# R reference for subsample / subset_data on the urchins dataset
suppressPackageStartupMessages(library(respR))
od <- 'X:/Rtools/resprpy/tests/reference'
urch <- read.csv(file.path(od, 'urchins.csv'), check.names = FALSE)

s1 <- subsample(urch, n = 100, plot = FALSE)
s2 <- subsample(urch, length.out = 50, plot = FALSE)
write.csv(data.frame(time = s1[[1]]), file.path(od, 'ref_subsample_n100.csv'), row.names = FALSE)
write.csv(data.frame(time = s2[[1]]), file.path(od, 'ref_subsample_len50.csv'), row.names = FALSE)

d1 <- subset_data(urch, from = 10, to = 30, by = 'time')
d2 <- subset_data(urch, from = 100, to = 300, by = 'row')
write.csv(d1, file.path(od, 'ref_subset_time.csv'), row.names = FALSE)
write.csv(d2, file.path(od, 'ref_subset_row.csv'), row.names = FALSE)
cat('subsample rows:', length(s1[[1]]), length(s2[[1]]),
    '| subset rows:', nrow(d1), nrow(d2), '\n')
