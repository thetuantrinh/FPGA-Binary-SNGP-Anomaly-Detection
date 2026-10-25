--------------------------------------------------------------------------------
-- tb_bsngp_core: streams STREAM_FILE (6 ints per line) into bsngp_core, one sample
-- every GAP cycles, and writes each result as "logit var abnormal unknown" to OUT_FILE.
-- sim/compare_rtl.py checks OUT_FILE against the golden model.
--------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;

entity tb_bsngp_core is
    generic (
        STREAM_FILE : string  := "stream.txt";
        OUT_FILE    : string  := "rtl_stream.txt";
        GAP         : natural := 80
    );
end entity tb_bsngp_core;

architecture sim of tb_bsngp_core is
    signal clk       : std_logic := '0';
    signal rst       : std_logic := '1';
    signal s_valid   : std_logic := '0';
    signal s_data    : std_logic_vector(47 downto 0) := (others => '0');
    signal res_valid, res_abn, res_unk, busy, overrun : std_logic;
    signal res_logit, res_var : std_logic_vector(31 downto 0);
    signal done      : boolean := false;
    signal n_results : natural := 0;
    signal max_busy  : natural := 0;
begin

    dut : entity work.bsngp_core
        port map (clk => clk, rst => rst, s_valid => s_valid, s_data => s_data,
                  res_valid => res_valid, res_abnormal => res_abn, res_unknown => res_unk,
                  res_logit => res_logit, res_var => res_var, busy => busy, overrun => overrun);

    p_clk : process
    begin
        while not done loop
            clk <= '0'; wait for 10 ns;
            clk <= '1'; wait for 10 ns;
        end loop;
        wait;
    end process;

    p_stim : process
        file f       : text open read_mode is STREAM_FILE;
        variable l   : line;
        variable v   : integer;
        variable d   : std_logic_vector(47 downto 0);
        variable n   : natural := 0;
        variable ol  : line;
    begin
        for i in 1 to 5 loop
            wait until rising_edge(clk);
        end loop;
        rst <= '0';
        while not endfile(f) loop
            readline(f, l);
            for i in 0 to 5 loop
                read(l, v);
                d(8*i+7 downto 8*i) := std_logic_vector(to_signed(v, 8));
            end loop;
            wait until rising_edge(clk);
            s_valid <= '1';
            s_data  <= d;
            wait until rising_edge(clk);
            s_valid <= '0';
            for g in 1 to GAP - 2 loop
                wait until rising_edge(clk);
            end loop;
            n := n + 1;
        end loop;
        for i in 1 to 4000 loop
            wait until rising_edge(clk);
        end loop;
        write(ol, string'("TB_CORE samples=")); write(ol, n);
        write(ol, string'(" results=")); write(ol, n_results);
        write(ol, string'(" overrun=")); write(ol, std_logic'image(overrun));
        write(ol, string'(" max_busy_cycles=")); write(ol, max_busy);
        writeline(output, ol);
        done <= true;
        wait;
    end process;

    p_cap : process
        file fo     : text open write_mode is OUT_FILE;
        variable l  : line;
        variable bc : natural := 0;
    begin
        while not done loop
            wait until rising_edge(clk) or done;
            if busy = '1' then
                bc := bc + 1;
            elsif bc > 0 then
                if bc > max_busy then
                    max_busy <= bc;
                end if;
                bc := 0;
            end if;
            if res_valid = '1' then
                write(l, to_integer(signed(res_logit))); write(l, string'(" "));
                write(l, to_integer(signed(res_var)));   write(l, string'(" "));
                if res_abn = '1' then write(l, string'("1 ")); else write(l, string'("0 ")); end if;
                if res_unk = '1' then write(l, string'("1")); else write(l, string'("0")); end if;
                writeline(fo, l);
                n_results <= n_results + 1;
            end if;
        end loop;
        wait;
    end process;

end architecture sim;
