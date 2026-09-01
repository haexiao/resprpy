# bench.R - speed benchmark: respR vs pyrespr (same data, saved to CSV)
suppressPackageStartupMessages(library(respR))
u <- read.csv("../tests/reference/urchins.csv")
mk <- function(n, seed) {
  set.seed(seed)
  t <- 0:(n - 1)
  o <- 8 - 0.001 * t + rnorm(n, 0, 0.01)
  data.frame(time = t, oxy = o)
}
d2k <- mk(2000, 1)
d10k <- mk(10000, 2)
write.csv(d2k, "bench_2k.csv", row.names = FALSE)
write.csv(d10k, "bench_10k.csv", row.names = FALSE)

bench <- function(label, expr) {
  tm <- system.time(expr)[3]
  cat(sprintf("%s %.4f\n", label, tm))
}

bench("calc_rate_271", calc_rate(u[, 1:2], from = 0, to = 45, by = "time",
                                 plot = FALSE))
bench("calc_rate_2k", calc_rate(d2k, from = 0, to = 1800, by = "time",
                                plot = FALSE))
bench("calc_rate_10k", calc_rate(d10k, from = 0, to = 9000, by = "time",
                                 plot = FALSE))
bench("auto_rate_271", auto_rate(u[, 1:2], plot = FALSE))
bench("auto_rate_2k", auto_rate(d2k, plot = FALSE))
bench("auto_rate_10k", auto_rate(d10k, plot = FALSE))
