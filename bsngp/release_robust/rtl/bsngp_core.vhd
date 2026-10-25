--------------------------------------------------------------------------------
-- bsngp_core: real-time Binary-SNGP anomaly + uncertainty accelerator
--
-- Implements bsngp/golden.py bit-exactly (verified by sim/run_sim.sh).
--
-- Streaming interface: one 6-channel int8 sample per s_valid pulse
--   s_data(8*i+7 downto 8*i) = channel i (ax, ay, az, gx, gy, gz), two's complement, |x| <= 127.
-- A window of the last 64 samples is evaluated after the 64th sample and then every
-- 32 samples. A trigger that arrives while the engine is busy sets the sticky
-- 'overrun' flag (cannot happen at 500 Hz / 50 MHz: ~1.8k cycles per window vs
-- 3.2M cycles between triggers).
--
-- Engine (sequential, ~1.8k cycles per window):
--   DC  (DC_REMOVE) subtract floor(window mean) per channel: one extra pass, 192 cycles
--   L1  binary conv 6->C1  (k=3,s=1,p=1), int8 x {+1,-1}: add/sub only
--   L2  binary conv C1->C2 (k=3,s=2,p=1), XNOR + popcount
--   L3  binary conv C2->C3 (k=3,s=2,p=1), XNOR + popcount, summed over time -> h
--   BN + sign folded into one integer comparison per channel (TH/DIR).
--   GP head: D random features, phase = (sum(+/-h) << KSH) + BR mod 1024 -> cos ROM,
--            logit L = BETA . phi + BETA0, variance V = |phi|^2 - sum_k (U_k . phi >> QSH)^2
-- Outputs: res_abnormal = L > TL, res_unknown = V > TV, plus raw L and V.
--------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.bsngp_params_pkg.all;

entity bsngp_core is
    port (
        clk          : in  std_logic;
        rst          : in  std_logic;  -- synchronous, active high
        s_valid      : in  std_logic;
        s_data       : in  std_logic_vector(47 downto 0);
        res_valid    : out std_logic;  -- one-cycle pulse
        res_abnormal : out std_logic;
        res_unknown  : out std_logic;
        res_logit    : out std_logic_vector(31 downto 0);
        res_var      : out std_logic_vector(31 downto 0);
        busy         : out std_logic;
        overrun      : out std_logic
    );
end entity bsngp_core;

architecture rtl of bsngp_core is

    constant WIN    : natural := 64;
    constant STRIDE : natural := 32;
    constant CMAX   : natural := 32;  -- >= C1, C2, C3

    -- Sample ring buffer (block RAM, registered read)
    type sbuf_t is array (0 to 127) of std_logic_vector(47 downto 0);
    signal sbuf     : sbuf_t;
    signal sb_raddr : unsigned(6 downto 0) := (others => '0');
    signal sb_q     : std_logic_vector(47 downto 0);
    signal wp       : unsigned(6 downto 0) := (others => '0');

    -- Feature maps between layers (block RAM / LUT RAM, registered read)
    type fm1_t is array (0 to 63) of std_logic_vector(C1-1 downto 0);
    type fm2_t is array (0 to 31) of std_logic_vector(C2-1 downto 0);
    signal fm1 : fm1_t;
    signal fm2 : fm2_t;
    signal fm1_we, fm2_we       : std_logic := '0';
    signal fm1_waddr, fm1_raddr : unsigned(5 downto 0) := (others => '0');
    signal fm2_waddr, fm2_raddr : unsigned(4 downto 0) := (others => '0');
    signal fm1_wdata, fm1_q     : std_logic_vector(C1-1 downto 0);
    signal fm2_wdata, fm2_q     : std_logic_vector(C2-1 downto 0);

    type state_t is (S_IDLE, S_MADDR, S_MWAIT, S_MACC, S_MDIV, S_CADDR, S_CWAIT, S_CACC, S_CTHR,
                     S_H0, S_H1, S_H2, S_H3, S_H4, S_V0, S_V1, S_DONE);
    signal state : state_t := S_IDLE;

    signal nfill  : integer range 0 to WIN := 0;
    signal phase  : integer range 0 to STRIDE-1 := 0;
    signal wstart : unsigned(6 downto 0) := (others => '0');

    signal layer  : integer range 1 to 3 := 1;
    signal t      : integer range 0 to 63 := 0;
    signal k      : integer range 0 to 2 := 0;
    signal tap_ok : std_logic := '0';

    -- Per-window channel means (DC removal)
    type msum_t is array (0 to 5) of signed(13 downto 0);
    signal msum : msum_t := (others => (others => '0'));
    type mean_t is array (0 to 5) of signed(8 downto 0);
    signal mean : mean_t := (others => (others => '0'));

    type acc_t is array (0 to CMAX-1) of signed(15 downto 0);
    signal acc : acc_t := (others => (others => '0'));
    type h_t is array (0 to C3-1) of signed(5 downto 0);
    signal h : h_t := (others => (others => '0'));

    -- GP head
    signal j     : integer range 0 to D-1 := 0;
    signal wrow  : std_logic_vector(C3-1 downto 0) := (others => '0');
    signal brj   : unsigned(9 downto 0) := (others => '0');
    signal betaj : signed(7 downto 0) := (others => '0');
    signal ucol_j : std_logic_vector(8*R-1 downto 0) := (others => '0');  -- U(:, j), 8 bits per row
    type psum_t is array (0 to 3) of signed(10 downto 0);
    signal psum  : psum_t := (others => (others => '0'));
    signal idx   : unsigned(9 downto 0) := (others => '0');
    signal phi   : signed(7 downto 0) := (others => '0');
    signal lacc  : signed(31 downto 0) := (others => '0');
    signal sacc  : signed(31 downto 0) := (others => '0');
    type p_t is array (0 to R-1) of signed(23 downto 0);
    signal pacc  : p_t := (others => (others => '0'));
    signal kk    : integer range 0 to R-1 := 0;
    signal vacc  : signed(31 downto 0) := (others => '0');

    signal overrun_r : std_logic := '0';

    -- sum_i (+/-) (x_i - mean_i) over the 6 input channels of one tap
    function l1_tap(q : std_logic_vector(47 downto 0); w : std_logic_vector; kt : integer; m : mean_t)
        return signed is
        variable s : signed(15 downto 0) := (others => '0');
        variable x : signed(15 downto 0);
    begin
        for i in 0 to 5 loop
            x := resize(signed(q(8*i+7 downto 8*i)), 16) - resize(m(i), 16);
            if w(w'low + kt*6 + i) = '1' then
                s := s + x;
            else
                s := s - x;
            end if;
        end loop;
        return s;
    end function;

    -- sum_i x_i * w_i for x, w in {+1 ('1'), -1 ('0')}: 2*popcount(xnor) - n
    -- (w holds all 3 taps; tap kt occupies bits kt*n .. kt*n+n-1, n = x'length)
    function pm_tap(x : std_logic_vector; w : std_logic_vector; kt : integer) return signed is
        variable pc : integer range 0 to 64 := 0;
        variable xv : std_logic_vector(x'length-1 downto 0) := x;
        variable wv : std_logic_vector(w'length-1 downto 0) := w;
    begin
        for i in 0 to x'length-1 loop
            if xv(i) = wv(kt*x'length + i) then
                pc := pc + 1;
            end if;
        end loop;
        return to_signed(2*pc - x'length, 16);
    end function;

    function thr(a : signed(15 downto 0); th : integer; ge : std_logic) return std_logic is
    begin
        if ge = '1' then
            if a >= to_signed(th, 16) then return '1'; else return '0'; end if;
        else
            if a <= to_signed(th, 16) then return '1'; else return '0'; end if;
        end if;
    end function;

begin

    -- Sample buffer: write on every incoming sample, registered read for the engine.
    p_sbuf : process(clk)
    begin
        if rising_edge(clk) then
            if s_valid = '1' then
                sbuf(to_integer(wp)) <= s_data;
            end if;
            sb_q <= sbuf(to_integer(sb_raddr));
        end if;
    end process;

    p_fm1 : process(clk)
    begin
        if rising_edge(clk) then
            if fm1_we = '1' then
                fm1(to_integer(fm1_waddr)) <= fm1_wdata;
            end if;
            fm1_q <= fm1(to_integer(fm1_raddr));
        end if;
    end process;

    p_fm2 : process(clk)
    begin
        if rising_edge(clk) then
            if fm2_we = '1' then
                fm2(to_integer(fm2_waddr)) <= fm2_wdata;
            end if;
            fm2_q <= fm2(to_integer(fm2_raddr));
        end if;
    end process;

    p_main : process(clk)
        variable trig  : boolean;
        variable pos   : integer range -1 to 127;
        variable lin   : integer range 0 to 64;
        variable lout  : integer range 0 to 64;
        variable str   : integer range 1 to 2;
        variable bits  : std_logic_vector(CMAX-1 downto 0);
        variable sv    : signed(10 downto 0);
        variable ph    : signed(15 downto 0);
        variable qv    : signed(15 downto 0);
    begin
        if rising_edge(clk) then
            res_valid <= '0';
            fm1_we    <= '0';
            fm2_we    <= '0';

            if rst = '1' then
                state     <= S_IDLE;
                wp        <= (others => '0');
                nfill     <= 0;
                phase     <= 0;
                overrun_r <= '0';
                res_abnormal <= '0';
                res_unknown  <= '0';
                res_logit    <= (others => '0');
                res_var      <= (others => '0');
            else
                ------------------------------------------------------------
                -- Sample counting and window trigger (runs while busy too)
                ------------------------------------------------------------
                trig := false;
                if s_valid = '1' then
                    wp <= wp + 1;
                    if nfill < WIN - 1 then
                        nfill <= nfill + 1;
                    elsif nfill = WIN - 1 then
                        nfill <= WIN;
                        phase <= 0;
                        trig  := true;
                    elsif phase = STRIDE - 1 then
                        phase <= 0;
                        trig  := true;
                    else
                        phase <= phase + 1;
                    end if;
                end if;

                if trig and state /= S_IDLE then
                    overrun_r <= '1';
                end if;

                ------------------------------------------------------------
                -- Layer geometry
                ------------------------------------------------------------
                case layer is
                    when 1      => lin := 64; lout := 64; str := 1;
                    when 2      => lin := 64; lout := 32; str := 2;
                    when others => lin := 32; lout := 16; str := 2;
                end case;

                case state is
                    when S_IDLE =>
                        if trig then
                            -- window = the 64 samples ending with the one written now (address wp)
                            wstart <= wp - to_unsigned(WIN - 1, 7);
                            layer  <= 1;
                            t      <= 0;
                            k      <= 0;
                            acc    <= (others => (others => '0'));
                            h      <= (others => (others => '0'));
                            msum   <= (others => (others => '0'));
                            mean   <= (others => (others => '0'));
                            if DC_REMOVE then
                                state <= S_MADDR;
                            else
                                state <= S_CADDR;
                            end if;
                        end if;

                    -- Window mean per channel (DC removal), 3 cycles per sample
                    when S_MADDR =>
                        sb_raddr <= wstart + to_unsigned(t, 7);
                        state    <= S_MWAIT;

                    when S_MWAIT =>
                        state <= S_MACC;

                    when S_MACC =>
                        for i in 0 to 5 loop
                            msum(i) <= msum(i) + resize(signed(sb_q(8*i+7 downto 8*i)), 14);
                        end loop;
                        if t = WIN - 1 then
                            t     <= 0;
                            state <= S_MDIV;
                        else
                            t     <= t + 1;
                            state <= S_MADDR;
                        end if;

                    when S_MDIV =>
                        for i in 0 to 5 loop
                            mean(i) <= resize(shift_right(msum(i), 6), 9);  -- floor(sum / 64)
                        end loop;
                        state <= S_CADDR;

                    when S_CADDR =>
                        pos := str * t + k - 1;
                        if pos >= 0 and pos < lin then
                            tap_ok <= '1';
                            case layer is
                                when 1      => sb_raddr  <= wstart + to_unsigned(pos, 7);
                                when 2      => fm1_raddr <= to_unsigned(pos, 6);
                                when others => fm2_raddr <= to_unsigned(pos, 5);
                            end case;
                        else
                            tap_ok <= '0';  -- zero padding
                        end if;
                        state <= S_CWAIT;

                    when S_CWAIT =>
                        state <= S_CACC;

                    when S_CACC =>
                        if tap_ok = '1' then
                            case layer is
                                when 1 =>
                                    for c in 0 to C1-1 loop
                                        acc(c) <= acc(c) + l1_tap(sb_q, W1(c), k, mean);
                                    end loop;
                                when 2 =>
                                    for c in 0 to C2-1 loop
                                        acc(c) <= acc(c) + pm_tap(fm1_q, W2(c), k);
                                    end loop;
                                when others =>
                                    for c in 0 to C3-1 loop
                                        acc(c) <= acc(c) + pm_tap(fm2_q, W3(c), k);
                                    end loop;
                            end case;
                        end if;
                        if k = 2 then
                            state <= S_CTHR;
                        else
                            k     <= k + 1;
                            state <= S_CADDR;
                        end if;

                    when S_CTHR =>
                        bits := (others => '0');
                        case layer is
                            when 1 =>
                                for c in 0 to C1-1 loop
                                    bits(c) := thr(acc(c), TH1(c), DIR1(c));
                                end loop;
                                fm1_we    <= '1';
                                fm1_waddr <= to_unsigned(t, 6);
                                fm1_wdata <= bits(C1-1 downto 0);
                            when 2 =>
                                for c in 0 to C2-1 loop
                                    bits(c) := thr(acc(c), TH2(c), DIR2(c));
                                end loop;
                                fm2_we    <= '1';
                                fm2_waddr <= to_unsigned(t, 5);
                                fm2_wdata <= bits(C2-1 downto 0);
                            when others =>
                                for c in 0 to C3-1 loop
                                    if thr(acc(c), TH3(c), DIR3(c)) = '1' then
                                        h(c) <= h(c) + 1;
                                    else
                                        h(c) <= h(c) - 1;
                                    end if;
                                end loop;
                        end case;
                        acc <= (others => (others => '0'));
                        k   <= 0;
                        if t = lout - 1 then
                            t <= 0;
                            if layer < 3 then
                                layer <= layer + 1;
                                state <= S_CADDR;
                            else
                                j     <= 0;
                                lacc  <= to_signed(BETA0, 32);
                                sacc  <= (others => '0');
                                pacc  <= (others => (others => '0'));
                                state <= S_H0;
                            end if;
                        else
                            t     <= t + 1;
                            state <= S_CADDR;
                        end if;

                    -- GP head: one random feature per 5 cycles
                    when S_H0 =>
                        wrow  <= WR(j);
                        brj   <= to_unsigned(BR(j), 10);
                        betaj <= to_signed(BETA(j), 8);
                        ucol_j <= UCOL(j);
                        state <= S_H1;

                    -- 32-term +/-h sum split into 4 registered partial sums (timing closure at 50 MHz);
                    -- the partial sums are combined in S_H2, so no cycle is added.
                    when S_H1 =>
                        for g in 0 to 3 loop
                            sv := (others => '0');
                            for c in g*(C3/4) to (g+1)*(C3/4)-1 loop
                                if wrow(c) = '1' then
                                    sv := sv + resize(h(c), 11);
                                else
                                    sv := sv - resize(h(c), 11);
                                end if;
                            end loop;
                            psum(g) <= sv;
                        end loop;
                        state <= S_H2;

                    when S_H2 =>
                        sv  := psum(0) + psum(1) + psum(2) + psum(3);
                        ph  := shift_left(resize(sv, 16), KSH) + signed(resize(brj, 16));
                        idx <= unsigned(ph(9 downto 0));  -- mod 1024
                        state <= S_H3;

                    when S_H3 =>
                        phi   <= to_signed(COS_LUT(to_integer(idx)), 8);
                        state <= S_H4;

                    when S_H4 =>
                        lacc <= lacc + resize(betaj * phi, 32);
                        sacc <= sacc + resize(phi * phi, 32);
                        for r_i in 0 to R-1 loop
                            pacc(r_i) <= pacc(r_i) + resize(signed(ucol_j(8*r_i+7 downto 8*r_i)) * phi, 24);
                        end loop;
                        if j = D-1 then
                            state <= S_V0;
                        else
                            j     <= j + 1;
                            state <= S_H0;
                        end if;

                    when S_V0 =>
                        vacc  <= sacc;
                        kk    <= 0;
                        state <= S_V1;

                    when S_V1 =>
                        qv   := resize(shift_right(pacc(kk), QSH), 16);
                        vacc <= vacc - resize(qv * qv, 32);
                        if kk = R-1 then
                            state <= S_DONE;
                        else
                            kk <= kk + 1;
                        end if;

                    when S_DONE =>
                        res_logit <= std_logic_vector(lacc);
                        res_var   <= std_logic_vector(vacc);
                        if lacc > to_signed(TL, 32) then
                            res_abnormal <= '1';
                        else
                            res_abnormal <= '0';
                        end if;
                        if vacc > to_signed(TV, 32) then
                            res_unknown <= '1';
                        else
                            res_unknown <= '0';
                        end if;
                        res_valid <= '1';
                        state     <= S_IDLE;
                end case;
            end if;
        end if;
    end process;

    busy    <= '0' when state = S_IDLE else '1';
    overrun <= overrun_r;

end architecture rtl;
