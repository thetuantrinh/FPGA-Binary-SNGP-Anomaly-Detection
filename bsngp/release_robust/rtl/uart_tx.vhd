-- Copied unchanged from de0_nano/uart_tx.vhd so bsngp/ is a self-contained Quartus project.
--------------------------------------------------------------------------------
-- Entity: uart_tx
-- Target: Terasic DE0-Nano (Altera Cyclone IV)
-- Description: Standard UART Transmitter (8-N-1) running on 50 MHz clock.
--              Default Baud: 115200 (50 MHz / 115200 = 434 clock cycles/bit)
--------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity uart_tx is
    generic (
        CLKS_PER_BIT : natural := 434 -- 50 MHz / 115200
    );
    port (
        clk       : in  std_logic;
        rst_n     : in  std_logic;
        tx_start  : in  std_logic;
        tx_byte   : in  std_logic_vector(7 downto 0);
        tx_serial : out std_logic;
        tx_busy   : out std_logic
    );
end entity uart_tx;

architecture rtl of uart_tx is

    type tx_state_t is (IDLE, START_BIT, DATA_BITS, STOP_BIT, CLEANUP);
    signal state : tx_state_t := IDLE;

    signal clk_count  : natural range 0 to CLKS_PER_BIT := 0;
    signal bit_index  : natural range 0 to 7 := 0;
    signal tx_data_reg: std_logic_vector(7 downto 0) := (others => '0');

begin

    process(clk, rst_n)
    begin
        if rst_n = '0' then
            state       <= IDLE;
            tx_serial   <= '1';
            tx_busy     <= '0';
            clk_count   <= 0;
            bit_index   <= 0;
            tx_data_reg <= (others => '0');

        elsif rising_edge(clk) then

            case state is
                when IDLE =>
                    tx_serial <= '1';
                    tx_busy   <= '0';
                    clk_count <= 0;
                    bit_index <= 0;

                    if tx_start = '1' then
                        tx_busy     <= '1';
                        tx_data_reg <= tx_byte;
                        state       <= START_BIT;
                    end if;

                when START_BIT =>
                    tx_busy   <= '1';
                    tx_serial <= '0'; -- Start bit is 0

                    if clk_count < CLKS_PER_BIT - 1 then
                        clk_count <= clk_count + 1;
                    else
                        clk_count <= 0;
                        state     <= DATA_BITS;
                    end if;

                when DATA_BITS =>
                    tx_busy   <= '1';
                    tx_serial <= tx_data_reg(bit_index);

                    if clk_count < CLKS_PER_BIT - 1 then
                        clk_count <= clk_count + 1;
                    else
                        clk_count <= 0;
                        if bit_index < 7 then
                            bit_index <= bit_index + 1;
                        else
                            bit_index <= 0;
                            state     <= STOP_BIT;
                        end if;
                    end if;

                when STOP_BIT =>
                    tx_busy   <= '1';
                    tx_serial <= '1'; -- Stop bit is 1

                    if clk_count < CLKS_PER_BIT - 1 then
                        clk_count <= clk_count + 1;
                    else
                        clk_count <= 0;
                        state     <= CLEANUP;
                    end if;

                when CLEANUP =>
                    tx_busy <= '0';
                    state   <= IDLE;

                when others =>
                    state <= IDLE;
            end case;

        end if;
    end process;

end architecture rtl;
