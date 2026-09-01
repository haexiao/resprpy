# R reference: calc_rate on the user's real OXY-10 data (files 1, 2, 6)
# using the same segments as their saved rmr1.csv
suppressPackageStartupMessages({library(respR); library(readxl)})
od <- 'X:/Rtools/resprpy/tests/reference'
datadir <- 'X:/Rtools/20260422/20260422 20'
seg <- read.csv(file.path(datadir, 'rmr1.csv'))   # from/to segments (time, sec)
files <- c(1, 2, 6)
for (f in files) {
  xlsx <- file.path(datadir, paste0(f, '.xlsx'))
  sheets <- excel_sheets(xlsx)
  chsheet <- sheets[grepl('Ch ', sheets)][1]
  d <- read_excel(xlsx, sheet = chsheet)
  df <- data.frame(time = as.numeric(d$`Delta T [min]`) * 60, oxygen = d$Oxygen)
  cr <- calc_rate(df, from = seg$time, to = seg$endtime, plot = FALSE)
  s <- cr$summary
  out <- data.frame(rank = s$rank, intercept_b0 = s$intercept_b0,
                    slope_b1 = s$slope_b1, rsq = s$rsq, row = s$row,
                    endrow = s$endrow, time = s$time, endtime = s$endtime,
                    oxy = s$oxy, endoxy = s$endoxy,
                    rate.2pt = s$rate.2pt, rate = s$rate)
  write.csv(out, file.path(od, paste0('ref_calc_rate_file', f, '.csv')), row.names = FALSE)
  cat('file', f, 'sheet', chsheet, 'nseg', nrow(out), 'ok\n')
}
