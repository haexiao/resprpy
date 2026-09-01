# R reference for the user's 0_RMR.R workflow (paths adapted to X:)
# Reads meas_params for 20260509 fish, applies permeability correction,
# calc_rate over all cycles, saves summary (columns 2:13, rate*3600).
suppressPackageStartupMessages({library(respR); library(lubridate); library(readxl)})
od <- 'X:/Rtools/resprpy/tests/reference'
wd <- 'X:/Rtools/20260422/20260509 5'
params_file <- 'X:/Rtools/20260422/raw/meas_params.csv'

meas_time <- 20260509
temperature <- 5
rmr_type <- 'fish'
channel <- 2:9

k <- matrix(data = NA, nrow = 9, ncol = 2)
colnames(k) <- c('channel', 'k_value')
k[, 'channel'] <- 1:9
k[, 'k_value'] <- c(0.0006223, 0.000317161, 0.001055724, 0.000671915,
                    0.000537423, 0.001256691, 0.000743536, 0.000743536, 0.000743536)

params <- read.csv(params_file, fileEncoding = 'UTF-8-BOM')
mode_params <- which(params$meas_time == meas_time & params$rmr_type == rmr_type)
cycles <- params$cycles[mode_params]
cycle_length <- params$cycle_length[mode_params]
initial <- params$initial[mode_params]
cycle_start <- params$cycle_start[mode_params]
cycle_time <- params$cycle_time[mode_params] - params$cycle_start[mode_params] - 5

cycle <- matrix(data = NA, nrow = cycles, ncol = 2)
colnames(cycle) <- c('start', 'end')
cycle[, 'start'] <- seq(from = cycle_start + initial, by = cycle_length, length.out = cycles)
cycle[, 'end'] <- cycle[, 'start'] + cycle_time

setwd(wd)
for (x in channel) {
  df <- read_excel(path = paste0(x, '.xlsx'), sheet = 6)
  data <- df[, c(2, 7)]
  names(data) <- c('Date', 'Oxygen')
  dtm <- parse_date_time(unlist(data$Date), 'ymdHMS')
  data$Date <- format(dtm, '%y/%m/%d %H:%M:%S')
  data <- format_time(data, format = 'ymdHMS')
  k_value <- k[k[, 'channel'] == x, 'k_value']
  max_oxy <- mean(df$Oxygen[order(df$Oxygen, decreasing = TRUE)[1:min(30, nrow(df))]], na.rm = TRUE)
  data$oxy <- data$Oxygen - k_value * (max_oxy - data$Oxygen)
  dataint <- inspect(data, time = 3, oxygen = 4)
  rates <- calc_rate(dataint, from = cycle[, 'start'], to = cycle[, 'end'], by = 'time')
  s <- rates$summary[, 2:13]
  s$rate <- s$rate * 3600
  write.csv(s, file.path(od, paste0('ref_0RMR_file', x, '.csv')), row.names = FALSE)
  cat('channel', x, 'cycles', nrow(s), 'ok\n')
}
