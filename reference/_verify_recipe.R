# Verify the user's calc_rate recipe against their saved rmr1.csv
suppressPackageStartupMessages({library(respR); library(readxl)})
d <- read_excel('X:/Rtools/20260422/20260422 20/1.xlsx', sheet = 'SAAU0002000027, Ch 1')
saveRDS(d, 'X:/Rtools/resprpy/tests/reference/ch1.rds')
# rmr1.csv row 1: row=157 endrow=432 time=781 endtime=2156
cat('nrow:', nrow(d), '\n')
cat('Id[157]:', d$Id[157], ' Id[432]:', d$Id[432], '\n')
cat('DeltaT[157]*60:', d$`Delta T [min]`[157]*60, ' DeltaT[432]*60:', d$`Delta T [min]`[432]*60, '\n')

# try recipe A: time = Id, oxygen = Oxygen
dfA <- data.frame(time = as.numeric(d$Id), oxygen = d$Oxygen)
crA <- calc_rate(dfA, from = 781, to = 2156, plot = FALSE)
cat('A row:', crA$summary$row, 'endrow:', crA$summary$endrow,
    'slope:', crA$summary$slope_b1, 'rsq:', crA$summary$rsq, '\n')

# try recipe B: time = Delta T [min]*60
dfB <- data.frame(time = as.numeric(d$`Delta T [min]`) * 60, oxygen = d$Oxygen)
crB <- calc_rate(dfB, from = 781, to = 2156, plot = FALSE)
cat('B row:', crB$summary$row, 'endrow:', crB$summary$endrow,
    'slope:', crB$summary$slope_b1, 'rsq:', crB$summary$rsq, '\n')
# rmr1.csv reference row 1: slope -4.31952799297228e-05, rsq 0.903, row 157, endrow 432
