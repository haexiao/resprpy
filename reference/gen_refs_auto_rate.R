# R reference for auto_rate on the urchins dataset (time col 1, oxygen col 15)
suppressPackageStartupMessages(library(respR))
od <- 'X:/Rtools/resprpy/tests/reference'
urch <- read.csv(file.path(od, 'urchins.csv'), check.names = FALSE)
df <- urch[, c(1, 15)]

ar <- auto_rate(df, plot = FALSE)
s <- ar$summary
write.csv(data.frame(rank = s$rank, intercept_b0 = s$intercept_b0, slope_b1 = s$slope_b1,
                     rsq = s$rsq, density = s$density, row = s$row, endrow = s$endrow,
                     time = s$time, endtime = s$endtime, oxy = s$oxy, endoxy = s$endoxy,
                     rate = s$rate),
          file.path(od, 'ref_auto_rate_urchins.csv'), row.names = FALSE)
d <- ar$density
write.csv(data.frame(x = d$x, y = d$y), file.path(od, 'ref_density_urchins.csv'), row.names = FALSE)
cat('bw:', d$bw, '| n:', nrow(ar$summary), '| roll:', nrow(ar$roll), '| peaks:', nrow(ar$peaks), '\n')

for (m in c('max', 'min', 'highest', 'lowest', 'interval', 'rolling')) {
  ar2 <- auto_rate(df, method = m, plot = FALSE)
  s2 <- ar2$summary
  write.csv(data.frame(rank = s2$rank, slope_b1 = s2$slope_b1, row = s2$row,
                       endrow = s2$endrow, time = s2$time, endtime = s2$endtime),
            file.path(od, paste0('ref_auto_rate_', m, '.csv')), row.names = FALSE)
}
cat('simple methods done\n')
