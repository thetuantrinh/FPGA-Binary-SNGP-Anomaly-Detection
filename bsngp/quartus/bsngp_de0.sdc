create_clock -name CLOCK_50 -period 20.000 [get_ports {CLOCK_50}]
derive_clock_uncertainty
# Asynchronous user inputs and UART pins are synchronised in the RTL.
set_false_path -from [get_ports {KEY[*] SW[*] UART_RXD}]
set_false_path -to   [get_ports {LED[*] UART_TXD}]
