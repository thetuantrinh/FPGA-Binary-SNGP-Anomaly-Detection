--------------------------------------------------------------------------------
-- bsngp_de0_top: Terasic DE0-Nano (EP4CE22F17C6) real-time Binary-SNGP demo
--
-- Sample sources (SW0):
--   SW0 = '1'  UART: PC/sensor streams frames  0x80, ax, ay, az, gx, gy, gz
--              (int8 two's complement, each in [-127, 127]; 0x80 is reserved as the sync byte)
--              115200 8N1 on UART_RXD (GPIO_0[0], JP1 pin 2). 7 bytes x 500 Hz = 3.5 kB/s.
--   SW0 = '0'  on-chip demo ROM replayed in a loop at 500 Hz (SW1 = '1': 50 Hz, slow-motion).
-- Result packet on UART_TXD (GPIO_0[1], JP1 pin 4) after every window (12 bytes):
--   0xA5, flags, seq, L[31:24], L[23:16], L[15:8], L[7:0], V[31:24], V[23:16], V[15:8], V[7:0], xor(bytes 1..10)
--   flags: bit0 abnormal, bit1 unknown (OOD), bit2 overrun (sticky), bit3 source (SW0)
-- LEDs: 0 abnormal, 1 unknown, 2 engine busy (stretched), 3 source = UART,
--       4 overrun, 5 new result (stretched), 6 UART frame received (stretched), 7 heartbeat
-- KEY0 = reset (active low). Changing SW0 restarts the window pipeline.
--------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.bsngp_demo_rom_pkg.all;

entity bsngp_de0_top is
    generic (
        CLKS_PER_BIT : natural := 434;     -- 50 MHz / 115200
        SAMPLE_DIV   : natural := 100000;  -- 50 MHz / 500 Hz
        STRETCH      : natural := 2500000  -- LED pulse stretch, 50 ms
    );
    port (
        CLOCK_50 : in  std_logic;
        KEY      : in  std_logic_vector(1 downto 0);
        SW       : in  std_logic_vector(3 downto 0);
        LED      : out std_logic_vector(7 downto 0);
        UART_RXD : in  std_logic;
        UART_TXD : out std_logic
    );
end entity bsngp_de0_top;

architecture rtl of bsngp_de0_top is

    signal clk : std_logic;
    signal rst_meta, rst_sync : std_logic := '1';
    signal rst_n : std_logic;
    signal sw_meta, sw_sync : std_logic_vector(1 downto 0) := (others => '0');
    signal src_uart, src_d : std_logic := '0';
    signal core_rst : std_logic := '1';

    -- UART RX + frame parser
    signal rx_valid : std_logic;
    signal rx_byte  : std_logic_vector(7 downto 0);
    signal fr_cnt   : integer range 0 to 6 := 0;  -- 0: waiting for sync
    signal fr_data  : std_logic_vector(39 downto 0) := (others => '0');
    signal u_valid  : std_logic := '0';
    signal u_data   : std_logic_vector(47 downto 0) := (others => '0');

    -- ROM replay
    signal tick_cnt : integer range 0 to 10*SAMPLE_DIV := 0;
    signal rom_addr : integer range 0 to DEMO_LEN-1 := 0;
    signal r_valid  : std_logic := '0';
    signal r_data   : std_logic_vector(47 downto 0) := (others => '0');

    -- Core
    signal s_valid : std_logic;
    signal s_data  : std_logic_vector(47 downto 0);
    signal res_valid, res_abn, res_unk, eng_busy, eng_overrun : std_logic;
    signal res_logit, res_var : std_logic_vector(31 downto 0);
    signal abn_r, unk_r : std_logic := '0';

    -- TX packetizer
    type pkt_t is array (0 to 11) of std_logic_vector(7 downto 0);
    signal pkt      : pkt_t := (others => (others => '0'));
    signal seq      : unsigned(7 downto 0) := (others => '0');
    type tx_state_t is (P_IDLE, P_START, P_WAIT_BUSY, P_WAIT_DONE);
    signal pstate   : tx_state_t := P_IDLE;
    signal pidx     : integer range 0 to 11 := 0;
    signal tx_start : std_logic := '0';
    signal tx_byte  : std_logic_vector(7 downto 0) := (others => '0');
    signal tx_busy  : std_logic;
    signal tx_ser   : std_logic;

    -- LEDs
    signal hb_cnt  : integer range 0 to 24999999 := 0;
    signal hb      : std_logic := '0';
    signal st_busy, st_res, st_rx : integer range 0 to STRETCH := 0;

begin

    clk <= CLOCK_50;

    p_sync : process(clk)
    begin
        if rising_edge(clk) then
            rst_meta <= not KEY(0);
            rst_sync <= rst_meta;
            sw_meta  <= SW(1 downto 0);
            sw_sync  <= sw_meta;
        end if;
    end process;
    rst_n    <= not rst_sync;
    src_uart <= sw_sync(0);

    -- Restart the window pipeline on reset or when the source changes.
    p_src : process(clk)
    begin
        if rising_edge(clk) then
            src_d    <= src_uart;
            core_rst <= rst_sync;
            if src_d /= src_uart then
                core_rst <= '1';
            end if;
        end if;
    end process;

    u_rx : entity work.uart_rx
        generic map (CLKS_PER_BIT => CLKS_PER_BIT)
        port map (clk => clk, rst_n => rst_n, rx_serial => UART_RXD, rx_valid => rx_valid, rx_byte => rx_byte);

    p_frame : process(clk)
    begin
        if rising_edge(clk) then
            u_valid <= '0';
            if rst_sync = '1' then
                fr_cnt <= 0;
            elsif rx_valid = '1' then
                if rx_byte = x"80" then
                    fr_cnt <= 1;                         -- sync byte (re)starts a frame
                elsif fr_cnt /= 0 then
                    case fr_cnt is
                        when 1      => fr_data(7 downto 0)   <= rx_byte;
                        when 2      => fr_data(15 downto 8)  <= rx_byte;
                        when 3      => fr_data(23 downto 16) <= rx_byte;
                        when 4      => fr_data(31 downto 24) <= rx_byte;
                        when 5      => fr_data(39 downto 32) <= rx_byte;
                        when others => null;  -- byte 6 goes straight to u_data
                    end case;
                    if fr_cnt = 6 then
                        u_valid <= '1';
                        u_data  <= rx_byte & fr_data(39 downto 0);
                        fr_cnt  <= 0;
                    else
                        fr_cnt <= fr_cnt + 1;
                    end if;
                end if;
            end if;
        end if;
    end process;

    p_rom : process(clk)
        variable div : integer range 0 to 10*SAMPLE_DIV;
    begin
        if rising_edge(clk) then
            r_valid <= '0';
            if sw_sync(1) = '1' then
                div := 10 * SAMPLE_DIV;
            else
                div := SAMPLE_DIV;
            end if;
            if core_rst = '1' then
                tick_cnt <= 0;
                rom_addr <= 0;
            elsif tick_cnt >= div - 1 then
                tick_cnt <= 0;
                r_valid  <= '1';
                r_data   <= DEMO_ROM(rom_addr);
                if rom_addr = DEMO_LEN - 1 then
                    rom_addr <= 0;
                else
                    rom_addr <= rom_addr + 1;
                end if;
            else
                tick_cnt <= tick_cnt + 1;
            end if;
        end if;
    end process;

    s_valid <= u_valid when src_uart = '1' else r_valid;
    s_data  <= u_data  when src_uart = '1' else r_data;

    u_core : entity work.bsngp_core
        port map (
            clk => clk, rst => core_rst,
            s_valid => s_valid, s_data => s_data,
            res_valid => res_valid, res_abnormal => res_abn, res_unknown => res_unk,
            res_logit => res_logit, res_var => res_var,
            busy => eng_busy, overrun => eng_overrun
        );

    -- Result packet transmitter
    p_tx : process(clk)
        variable flags : std_logic_vector(7 downto 0);
        variable chk   : std_logic_vector(7 downto 0);
    begin
        if rising_edge(clk) then
            tx_start <= '0';
            if rst_sync = '1' then
                pstate <= P_IDLE;
                seq    <= (others => '0');
                abn_r  <= '0';
                unk_r  <= '0';
            else
                if res_valid = '1' then
                    abn_r <= res_abn;
                    unk_r <= res_unk;
                end if;
                case pstate is
                    when P_IDLE =>
                        if res_valid = '1' then
                            flags := "0000" & src_uart & eng_overrun & res_unk & res_abn;
                            pkt(0)  <= x"A5";
                            pkt(1)  <= flags;
                            pkt(2)  <= std_logic_vector(seq);
                            pkt(3)  <= res_logit(31 downto 24);
                            pkt(4)  <= res_logit(23 downto 16);
                            pkt(5)  <= res_logit(15 downto 8);
                            pkt(6)  <= res_logit(7 downto 0);
                            pkt(7)  <= res_var(31 downto 24);
                            pkt(8)  <= res_var(23 downto 16);
                            pkt(9)  <= res_var(15 downto 8);
                            pkt(10) <= res_var(7 downto 0);
                            chk := flags xor std_logic_vector(seq)
                                   xor res_logit(31 downto 24) xor res_logit(23 downto 16)
                                   xor res_logit(15 downto 8) xor res_logit(7 downto 0)
                                   xor res_var(31 downto 24) xor res_var(23 downto 16)
                                   xor res_var(15 downto 8) xor res_var(7 downto 0);
                            pkt(11) <= chk;
                            seq     <= seq + 1;
                            pidx    <= 0;
                            pstate  <= P_START;
                        end if;
                    when P_START =>
                        tx_byte  <= pkt(pidx);
                        tx_start <= '1';
                        pstate   <= P_WAIT_BUSY;
                    when P_WAIT_BUSY =>
                        if tx_busy = '1' then
                            pstate <= P_WAIT_DONE;
                        end if;
                    when P_WAIT_DONE =>
                        if tx_busy = '0' then
                            if pidx = 11 then
                                pstate <= P_IDLE;
                            else
                                pidx   <= pidx + 1;
                                pstate <= P_START;
                            end if;
                        end if;
                end case;
            end if;
        end if;
    end process;

    u_tx : entity work.uart_tx
        generic map (CLKS_PER_BIT => CLKS_PER_BIT)
        port map (clk => clk, rst_n => rst_n, tx_start => tx_start, tx_byte => tx_byte,
                  tx_serial => tx_ser, tx_busy => tx_busy);
    UART_TXD <= tx_ser;

    p_led : process(clk)
    begin
        if rising_edge(clk) then
            if hb_cnt = 24999999 then
                hb_cnt <= 0;
                hb     <= not hb;
            else
                hb_cnt <= hb_cnt + 1;
            end if;
            if eng_busy = '1' then st_busy <= STRETCH; elsif st_busy > 0 then st_busy <= st_busy - 1; end if;
            if res_valid = '1' then st_res <= STRETCH; elsif st_res > 0 then st_res <= st_res - 1; end if;
            if u_valid = '1' then st_rx <= STRETCH; elsif st_rx > 0 then st_rx <= st_rx - 1; end if;
        end if;
    end process;

    LED(0) <= abn_r;
    LED(1) <= unk_r;
    LED(2) <= '1' when st_busy > 0 else '0';
    LED(3) <= src_uart;
    LED(4) <= eng_overrun;
    LED(5) <= '1' when st_res > 0 else '0';
    LED(6) <= '1' when st_rx > 0 else '0';
    LED(7) <= hb;

end architecture rtl;
