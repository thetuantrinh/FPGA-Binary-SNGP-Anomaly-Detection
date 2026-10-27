--------------------------------------------------------------------------------
-- tb_bsngp_top: end-to-end test of bsngp_de0_top through its real pins.
--   MODE = 1: drives UART frames (0x80 + 6 int8) bit by bit from STREAM_FILE.
--   MODE = 0: stand-alone ROM replay (SW0 = '0').
-- Decodes the 12-byte result packets from UART_TXD, checks sync and checksum, and
-- writes "logit var abnormal unknown" per packet to OUT_FILE.
-- Fast settings for simulation: CLKS_PER_BIT = 8, SAMPLE_DIV = 100.
--------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;

entity tb_bsngp_top is
    generic (
        MODE        : natural := 1;
        STREAM_FILE : string  := "stream.txt";
        OUT_FILE    : string  := "rtl_top.txt";
        N_SAMPLES   : natural := 256;   -- MODE 1
        N_RESULTS   : natural := 7
    );
end entity tb_bsngp_top;

architecture sim of tb_bsngp_top is
    constant CPB : natural := 8;
    signal clk  : std_logic := '0';
    signal key  : std_logic_vector(1 downto 0) := "10";
    signal sw   : std_logic_vector(3 downto 0) := (others => '0');
    signal led  : std_logic_vector(7 downto 0);
    signal rxd  : std_logic := '1';
    signal txd  : std_logic;
    signal done : boolean := false;
    signal n_pkts, n_bad : natural := 0;
begin

    dut : entity work.bsngp_de0_top
        generic map (CLKS_PER_BIT => CPB, SAMPLE_DIV => 100, STRETCH => 10)
        port map (CLOCK_50 => clk, KEY => key, SW => sw, LED => led, UART_RXD => rxd, UART_TXD => txd);

    p_clk : process
    begin
        while not done loop
            clk <= '0'; wait for 10 ns;
            clk <= '1'; wait for 10 ns;
        end loop;
        wait;
    end process;

    p_stim : process
        file f      : text open read_mode is STREAM_FILE;
        variable l  : line;
        variable v  : integer;
        variable ol : line;

        procedure tick(n : natural) is
        begin
            for i in 1 to n loop
                wait until rising_edge(clk);
            end loop;
        end procedure;

        procedure send_byte(b : std_logic_vector(7 downto 0)) is
        begin
            rxd <= '0'; tick(CPB);
            for i in 0 to 7 loop
                rxd <= b(i); tick(CPB);
            end loop;
            rxd <= '1'; tick(2 * CPB);
        end procedure;
    begin
        if MODE = 1 then sw(0) <= '1'; else sw(0) <= '0'; end if;
        key(0) <= '0';
        tick(20);
        key(0) <= '1';
        tick(20);
        if MODE = 1 then
            for s in 1 to N_SAMPLES loop
                exit when endfile(f);
                readline(f, l);
                send_byte(x"80");
                for i in 0 to 5 loop
                    read(l, v);
                    send_byte(std_logic_vector(to_signed(v, 8)));
                end loop;
            end loop;
        end if;
        -- wait for the remaining packets
        for i in 1 to 400000 loop
            exit when n_pkts >= N_RESULTS;
            wait until rising_edge(clk);
        end loop;
        tick(2000);
        write(ol, string'("TB_TOP mode=")); write(ol, MODE);
        write(ol, string'(" packets=")); write(ol, n_pkts);
        write(ol, string'(" bad_packets=")); write(ol, n_bad);
        write(ol, string'(" overrun_led=")); write(ol, std_logic'image(led(4)));
        writeline(output, ol);
        done <= true;
        wait;
    end process;

    -- UART receiver + packet decoder on the DUT's TX pin
    p_rx : process
        file fo      : text open write_mode is OUT_FILE;
        variable l   : line;
        type pkt_t is array (0 to 11) of std_logic_vector(7 downto 0);
        variable pkt : pkt_t;
        variable b   : std_logic_vector(7 downto 0);
        variable idx : natural := 0;
        variable chk : std_logic_vector(7 downto 0);
        variable lg, vr : signed(31 downto 0);
    begin
        while not done loop
            wait until txd = '0' or done;
            exit when done;
            for i in 1 to CPB / 2 loop wait until rising_edge(clk); end loop;   -- middle of start bit
            for i in 0 to 7 loop
                for c in 1 to CPB loop wait until rising_edge(clk); end loop;
                b(i) := txd;
            end loop;
            for c in 1 to CPB loop wait until rising_edge(clk); end loop;      -- stop bit
            if idx = 0 and b /= x"A5" then
                null;  -- resync
            else
                pkt(idx) := b;
                if idx = 11 then
                    chk := (others => '0');
                    for i in 1 to 10 loop chk := chk xor pkt(i); end loop;
                    if chk /= pkt(11) then
                        n_bad <= n_bad + 1;
                    end if;
                    lg := signed(std_logic_vector'(pkt(3) & pkt(4) & pkt(5) & pkt(6)));
                    vr := signed(std_logic_vector'(pkt(7) & pkt(8) & pkt(9) & pkt(10)));
                    write(l, to_integer(lg)); write(l, string'(" "));
                    write(l, to_integer(vr)); write(l, string'(" "));
                    if pkt(1)(0) = '1' then write(l, string'("1 ")); else write(l, string'("0 ")); end if;
                    if pkt(1)(1) = '1' then write(l, string'("1")); else write(l, string'("0")); end if;
                    writeline(fo, l);
                    n_pkts <= n_pkts + 1;
                    idx := 0;
                else
                    idx := idx + 1;
                end if;
            end if;
        end loop;
        wait;
    end process;

end architecture sim;
