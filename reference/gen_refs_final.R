# gen_refs_final.R - generate R reference CSVs for the final batch of
# ported functions: calc_rate.int/bg/ft, auto_rate.int, adjust_rate,
# select_rate, oxy_crit, inspect, convert_rate.ft, test_lin.
# Run with: Rscript gen_refs_final.R   (ASCII only; run from reference/)
suppressPackageStartupMessages(library(respR))
suppressPackageStartupMessages(library(data.table))

dir.create("../tests/reference", showWarnings = FALSE, recursive = TRUE)
u <- read.csv("../tests/reference/urchins.csv")

wr <- function(obj, name, path = "../tests/reference") {
  cat("writing:", name, "\n")
  write.csv(obj, file = file.path(path, name), row.names = FALSE)
}

# ---- calc_rate.int (by row) -----------------------------------------------
cri <- calc_rate.int(u, starts = 50, wait = 10, measure = 80, by = "row",
                     plot = FALSE)
s <- as.data.frame(cri$summary)
s$rate <- cri$rate
wr(s, "ref_calc_rate_int_row.csv")

cri2 <- calc_rate.int(u, starts = 20, wait = 2, measure = 15, by = "time",
                      plot = FALSE)
s2 <- as.data.frame(cri2$summary)
s2$rate <- cri2$rate
wr(s2, "ref_calc_rate_int_time.csv")

# ---- calc_rate.bg ----------------------------------------------------------
crb <- calc_rate.bg(u[, 1:3], time = 1, oxygen = c(2, 3), plot = FALSE)
sb <- as.data.frame(crb$summary)
sb$rate.bg.mean <- crb$rate.bg.mean
wr(sb, "ref_calc_rate_bg.csv")

# ---- calc_rate.ft (numeric vector + rolling width on a data frame) ---------
set.seed(42)
delta <- runif(100, 0.005, 0.02)
wr(data.frame(delta = delta), "ref_ft_delta.csv")
crf <- calc_rate.ft(delta, flowrate = 0.1, plot = FALSE)
sf <- as.data.frame(crf$summary)
wr(sf, "ref_calc_rate_ft_vec.csv")

# 2-col df input (out, in)
set.seed(43)
outo <- 8 - seq(0, 0.5, length.out = 100) + rnorm(100, 0, 0.01)
ino <- rep(7.5, 100)
wr(data.frame(out = outo, ino = ino), "ref_ft_input.csv")
crf2 <- calc_rate.ft(cbind(outo, ino), flowrate = 0.05, plot = FALSE)
sf2 <- as.data.frame(crf2$summary)
wr(sf2, "ref_calc_rate_ft_df.csv")

# inspect.ft -> calc_rate.ft
ift <- inspect.ft(data.frame(time = seq(0, 100, length.out = 100),
                             out = outo, ino = ino), time = 1,
                  out.oxy = 2, in.oxy = 3, plot = FALSE)
crf3 <- calc_rate.ft(ift, flowrate = 0.05, from = 0, to = 100, by = "time",
                     plot = FALSE)
sf3 <- as.data.frame(crf3$summary)
wr(sf3, "ref_calc_rate_ft_insp.csv")

# rolling width branch
crf4 <- calc_rate.ft(ift, flowrate = 0.05, by = "row", width = 20,
                     plot = FALSE)
sf4 <- as.data.frame(crf4$summary)
wr(sf4, "ref_calc_rate_ft_width.csv")

# ---- auto_rate.int ---------------------------------------------------------
ari <- auto_rate.int(u, starts = 32, wait = 5, measure = 20, by = "row",
                     method = "linear", width = 8, plot = FALSE)
sa <- as.data.frame(ari$summary)
wr(sa, "ref_auto_rate_int.csv")

# ---- adjust_rate -----------------------------------------------------------
set.seed(7)
t2 <- seq(0, 300, by = 2)
o2 <- 8 - 0.0008 * t2 + rnorm(length(t2), 0, 0.005)
dat2 <- data.frame(time = t2, oxy = o2)
wr(dat2, "ref_adjust_dat2.csv")
cr <- calc_rate(dat2, from = c(0, 100, 200), to = c(80, 180, 280),
                by = "time", plot = FALSE)
bgv <- calc_rate.bg(dat2, time = 1, oxygen = 2, plot = FALSE)

adj1 <- adjust_rate(cr, by = -0.0002, method = "value")
sa1 <- as.data.frame(adj1$summary)
wr(sa1, "ref_adjust_rate_value.csv")

adj2 <- adjust_rate(cr, by = bgv, method = "mean")
sa2 <- as.data.frame(adj2$summary)
wr(sa2, "ref_adjust_rate_mean.csv")

adj3 <- adjust_rate(cr, by = rep(-0.0001, 3), method = "paired")
sa3 <- as.data.frame(adj3$summary)
wr(sa3, "ref_adjust_rate_paired.csv")

bgdf1 <- data.frame(time = seq(0, 60, by = 2), oxy = 7 - 0.001 * seq(0, 60, by = 2))
bgdf2 <- data.frame(time = seq(240, 300, by = 2), oxy = 6.7 - 0.002 * seq(0, 60, by = 2))
adj4 <- adjust_rate(cr, by = bgdf1, by2 = bgdf2, method = "linear")
sa4 <- as.data.frame(adj4$summary)
wr(sa4, "ref_adjust_rate_linear.csv")

adj5 <- adjust_rate(cr, by = bgdf1, by2 = bgdf2, method = "exponential")
sa5 <- as.data.frame(adj5$summary)
wr(sa5, "ref_adjust_rate_exp.csv")

# concurrent
adj6 <- adjust_rate(cr, by = dat2, method = "concurrent")
sa6 <- as.data.frame(adj6$summary)
wr(sa6, "ref_adjust_rate_concurrent.csv")

# ---- select_rate -----------------------------------------------------------
cv <- convert_rate(cr, oxy.unit = "mg/L", time.unit = "sec",
                   output.unit = "mg/h/kg", volume = 0.1, mass = 0.01,
                   quiet = TRUE)
sr <- select_rate(cv, "minimum", n = 2)
so <- sr$summary
so$rate.output <- sr$rate.output
wr(as.data.frame(so), "ref_select_rate_min.csv")

sr2 <- select_rate(cv, "maximum")
so2 <- sr2$summary
so2$rate.output <- sr2$rate.output
wr(as.data.frame(so2), "ref_select_rate_max.csv")

sr3 <- select_rate(cv, "rate", n = c(-100, 100))
so3 <- sr3$summary
so3$rate.output <- sr3$rate.output
wr(as.data.frame(so3), "ref_select_rate_range.csv")

# ---- oxy_crit --------------------------------------------------------------
oc <- oxy_crit(u, method = "bsr", plot = FALSE)
ro <- as.data.frame(oc$results)
critdf <- data.frame(crit.intercept = oc$crit$crit.intercept,
                     crit.midpoint = oc$crit$crit.midpoint)
wr(ro, "ref_oxy_crit_results.csv")
wr(critdf, "ref_oxy_crit_crit.csv")

# ---- inspect ---------------------------------------------------------------
ins <- inspect(u[, 1:2], time = 1, oxygen = 2, plot = FALSE)
checks <- as.data.frame(lapply(as.data.frame(ins$checks), as.character))
wr(checks, "ref_inspect_checks.csv")
locs_df <- data.frame(locs = sapply(ins$locs, function(z) paste(z, collapse = ",")))
wr(locs_df, "ref_inspect_locs.csv")

# inspect with issues
bad <- data.frame(time = c(1, 2, NA, 4, 4, 3), oxy = c(8, 7.9, 7.8, Inf, 7.6, 7.5))
ins2 <- inspect(bad, time = 1, oxygen = 2, plot = FALSE)
wr(as.data.frame(lapply(as.data.frame(ins2$checks), as.character)),
   "ref_inspect_bad_checks.csv")

# inspect.ft
insf <- inspect.ft(data.frame(time = seq(0, 100, length.out = 100),
                              out = outo, ino = ino), time = 1,
                   out.oxy = 2, in.oxy = 3, plot = FALSE)
wr(as.data.frame(lapply(as.data.frame(insf$checks), as.character)),
   "ref_inspect_ft_checks.csv")
dfo <- as.data.frame(insf$dataframe)
wr(dfo, "ref_inspect_ft_dataframe.csv")

# ---- convert_rate.ft -------------------------------------------------------
cft <- convert_rate.ft(c(0.01, 0.012, 0.008), oxy.unit = "mg/L",
                       flowrate.unit = "mL/min", output.unit = "mg/h",
                       plot = FALSE)
sc <- as.data.frame(cft$summary)
wr(sc, "ref_convert_rate_ft_vec.csv")

cft2 <- convert_rate.ft(crf3, oxy.unit = "mg/L", flowrate.unit = "mL/min",
                        output.unit = "mg/h/kg", mass = 0.05, plot = FALSE)
sc2 <- as.data.frame(cft2$summary)
wr(sc2, "ref_convert_rate_ft_obj.csv")

cft3 <- convert_rate.ft(c(0.01, 0.012), oxy.unit = "mg/L",
                        flowrate.unit = "mL/min", output.unit = "mg/h/m2",
                        area = 0.02, plot = FALSE)
sc3 <- as.data.frame(cft3$summary)
wr(sc3, "ref_convert_rate_ft_area.csv")

# ---- test_lin (structure only; RNG differs between R and numpy) ------------
set.seed(99)
tl <- test_lin(reps = 5, plot = FALSE)
wr(as.data.frame(tl$df), "ref_test_lin_df.csv")
wr(data.frame(coef_int = coef(tl$results)[1], coef_slp = coef(tl$results)[2],
              rsq = summary(tl$results)$r.squared), "ref_test_lin_lm.csv")

cat("ALL REFS WRITTEN\n")
