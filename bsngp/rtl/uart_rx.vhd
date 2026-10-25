-- Copied unchanged from de0_nano/uart_rx.vhd so bsngp/ is a self-contained Quartus project.
--------------------------------------------------------------------------------
-- Entity: uart_rx
-- Target: Terasic DE0-Nano (Altera Cyclone IV)
-- Description: Standard UART Receiver (8-N-1) running on 50 MHz clock.
--              Default Baud: 115200 (50 MHz / 115200 = 434 clock cycles/bit)
--------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity uart_rx is
    generic (
        CLKS_PER_BIT : natural := 434 -- 50 MHz / 115200
    );
    port (
        clk       : in  std_logic;
        rst_n     : in  std_logic;
        rx_serial : in  std_logic;
        rx_valid  : out std_logic;
        rx_byte   : out std_logic_vector(7 downto 0)
    );
end entity uart_rx;

architecture rtl of uart_rx is

    type rx_state_t is (IDLE, START_BIT, DATA_BITS, STOP_BIT, CLEANUP);
    signal state : rx_state_t := IDLE;

    signal rx_sync_d1 : std_logic := '1';
    signal rx_sync    : std_logic := '1';

    signal clk_count  : natural range 0 to CLKS_PER_BIT := 0;
    signal bit_index  : natural range 0 to 7 := 0;
    signal rx_shift   : std_logic_vector(7 downto 0) := (others => '0');

begin

    process(clk, rst_n)
    begin
        if rst_n = '0' then
            state      <= IDLE;
            rx_valid   <= '0';
            rx_byte    <= (others => '0');
            clk_count  <= 0;
            bit_index  <= 0;
            rx_sync_d1 <= '1';
            rx_sync    <= '1';

        elsif rising_edge(clk) then
            -- Double-register input to prevent metastability
            rx_sync_d1 <= rx_serial;
            rx_sync    <= rx_sync_d1;
            rx_valid   <= '0';

            case state is
                when IDLE =>
                    clk_count <= 0;
                    bit_index <= 0;
                    if rx_sync = '0' then -- Falling edge: Start bit detected
                        state <= START_BIT;
                    end if;

                when START_BIT =>
                    -- Check middle of start bit
                    if clk_count = (CLKS_PER_BIT - 1) / 2 then
                        if rx_sync = '0' then
                            clk_count <= 0;
                            state     <= DATA_BITS;
                        else
                            state     <= IDLE; -- False start glitch
                        end if;
                    else
                        clk_count <= clk_count + 1;
                    end if;

                when DATA_BITS =>
                    if clk_count < CLKS_PER_BIT - 1 then
                        clk_count <= clk_count + 1;
                    else
                        clk_count <= 0;
                        rx_shift(bit_index) <= rx_sync;
                        if bit_index < 7 then
                            bit_index <= bit_index + 1;
                        else
                            bit_index <= 0;
                            state     <= STOP_BIT;
                        end if;
                    end if;

                when STOP_BIT =>
                    if clk_count < CLKS_PER_BIT - 1 then
                        clk_count <= clk_count + 1;
                    else
                        clk_count <= 0;
                        rx_valid  <= '1';
                        rx_byte   <= rx_shift;
                        state     <= CLEANUP;
                    end if;

                when CLEANUP =>
                    state <= IDLE;

                when others =>
                    state <= IDLE;
            end case;
        end if;
    end process;

end architecture rtl;
