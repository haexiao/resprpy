# gen_refs_select_all.R - select_rate all-methods reference (R 4.5.3 + respR)
suppressPackageStartupMessages(library(respR))
wr <- function(x, nm) write.csv(x, paste0("../tests/reference/", nm),
                                row.names = FALSE)
set.seed(7)
t2 <- seq(0, 300, by = 2)
o2 <- 8 - 0.0008 * t2 + rnorm(length(t2), 0, 0.005)
dat2 <- data.frame(time = t2, oxy = o2)
cr <- calc_rate(dat2, from = c(0, 100, 200), to = c(80, 180, 280),
                by = "time", plot = FALSE)
cv <- convert_rate(cr, oxy.unit = "mg/L", time.unit = "sec",
                   output.unit = "mg/h/kg", volume = 0.1, mass = 0.01,
                   quiet = TRUE)
num_cols <- c("rep", "rank", "intercept_b0", "slope_b1", "rsq", "density",
              "row", "endrow", "time", "endtime", "oxy", "endoxy", "rate",
              "adjustment", "rate.adjusted", "rate.input", "volume", "mass",
              "area", "S", "t", "P", "rate.abs", "rate.m.spec",
              "rate.a.spec", "rate.output")

methods <- c("overlap", "duration", "density", "manual", "manual_omit",
             "time", "time_omit", "row", "row_omit", "oxygen", "oxygen_omit",
             "rsq", "intercept", "slope", "rep", "rep_omit", "rank",
             "rank_omit", "rate", "rate.output", "maximum_percentile",
             "minimum_percentile", "maximum", "minimum", "lowest_percentile",
             "highest_percentile", "lowest", "highest", "zero", "nonzero",
             "negative", "positive", "rolling", "linear")
nvals <- list(
  overlap = 0, duration = c(0, 10), density = NULL, manual = 1,
  manual_omit = 1, time = c(0, 80), time_omit = c(0, 80), row = c(1, 40),
  row_omit = c(1, 40), oxygen = c(6, 8), oxygen_omit = c(6, 8),
  rsq = c(0.5, 1), intercept = c(-1, 1), slope = c(-1, 1), rep = 1,
  rep_omit = 1, rank = 1, rank_omit = 1, rate = c(-1, 1),
  rate.output = c(-1, 1), maximum_percentile = 0.5,
  minimum_percentile = 0.5, maximum = 1, minimum = 1,
  lowest_percentile = 0.5, highest_percentile = 0.5, lowest = 1,
  highest = 1, zero = NULL, nonzero = NULL, negative = NULL,
  positive = NULL, rolling = 1, linear = NULL)

status <- character(0)
for (m in methods) {
  out <- tryCatch({
    sr <- select_rate(cv, method = m, n = nvals[[m]])
    s <- as.data.frame(sr$summary)
    s <- s[, num_cols[num_cols %in% names(s)], drop = FALSE]
    wr(s, paste0("ref_select_all_", m, ".csv"))
    "OK"
  }, error = function(e) paste0("ERR:", substr(conditionMessage(e), 1, 60)))
  status[m] <- out
  cat(m, "->", out, "\n")
}
write.csv(data.frame(method = names(status), status = status),
          "../tests/reference/ref_select_all_status.csv", row.names = FALSE)
cat("DONE\n")
