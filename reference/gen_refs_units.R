# Generate R reference outputs for the ported modules (units/marelac/convert).
# Run with: Rscript reference/gen_refs_units.R
suppressPackageStartupMessages(library(respR))
od <- "X:/Rtools/resprpy/tests/reference"
urch <- read.csv(file.path(od, "urchins.csv"), check.names = FALSE)

# ---------- 1. marelac physical anchors ----------
S <- c(0, 35, 35); t <- c(20, 20, 25); P <- c(1.013253, 1.013253, 1.0)
write.csv(data.frame(
  vapor = marelac::vapor(S = S, t = t),
  gas_solubility = marelac::gas_solubility(S = S, t = t, species = "O2"),
  gas_satconc = marelac::gas_satconc(S = S, t = t, P = P, species = "O2"),
  molvol = marelac::molvol(t = t, P = P, species = "O2"),
  sw_dens = marelac::sw_dens(S = S, t = t, P = P),
  molweight = rep(marelac::molweight("O2"), 3)
), file.path(od, "ref_marelac.csv"), row.names = FALSE)

# ---------- 2. convert_DO on urchins oxygen column (col 15) ----------
oxy <- urch[, 15]
write.csv(data.frame(
  umol.L  = convert_DO(oxy, "mg/L", "umol/L",  S = 35, t = 20, P = 1.013253),
  kPa     = convert_DO(oxy, "mg/L", "kPa",     S = 35, t = 20, P = 1.013253),
  pctAir  = convert_DO(oxy, "mg/L", "%Air",    S = 35, t = 20, P = 1.013253),
  mL.L    = convert_DO(oxy, "mg/L", "mL/L",    S = 35, t = 20, P = 1.013253),
  Torr    = convert_DO(oxy, "mg/L", "Torr",    S = 35, t = 20, P = 1.013253),
  mg.kg   = convert_DO(oxy, "mg/L", "mg/kg",   S = 35, t = 20, P = 1.013253),
  umol.kg = convert_DO(oxy, "mg/L", "umol/kg", S = 35, t = 20, P = 1.013253)
), file.path(od, "ref_convert_DO.csv"), row.names = FALSE)

# ---------- 3. convert_val ----------
write.csv(data.frame(
  vol      = convert_val(c(1, 2.5, 0.05), "L", "mL"),
  tempCtoK = convert_val(c(0, 25, 100),   "C", "K"),
  tempFtoC = convert_val(c(32, 98.6, 212), "F", "C"),
  mass     = convert_val(c(100, 2500, 5), "g", "kg"),
  area     = convert_val(c(1, 2.5, 0.01), "m2", "cm2"),
  pres     = convert_val(c(1, 1.013253, 0.5), "bar", "kPa")
), file.path(od, "ref_convert_val.csv"), row.names = FALSE)

# ---------- 4. convert_rate (numeric input) ----------
rates <- c(-0.001, -0.002, -0.0005)
r1 <- suppressMessages(suppressWarnings(convert_rate(
  rates, oxy.unit = "mg/L", time.unit = "sec", output.unit = "mg/h",
  volume = 0.6, S = 35, t = 20, P = 1.013253)))
r2 <- suppressMessages(suppressWarnings(convert_rate(
  rates, oxy.unit = "mg/L", time.unit = "sec", output.unit = "mg/h/kg",
  volume = 0.6, mass = 0.4, S = 35, t = 20, P = 1.013253)))
r3 <- suppressMessages(suppressWarnings(convert_rate(
  rates, oxy.unit = "mg/L", time.unit = "min", output.unit = "umol/min/g",
  volume = 0.6, mass = 0.4, S = 35, t = 20, P = 1.013253)))
r4 <- suppressMessages(suppressWarnings(convert_rate(
  rates, oxy.unit = "mg/L", time.unit = "hour", output.unit = "mL/h/m2",
  volume = 0.6, area = 0.01, S = 35, t = 20, P = 1.013253)))
write.csv(data.frame(
  mg.h = r1$rate.output, mg.h.kg = r2$rate.output,
  umol.min.g = r3$rate.output, mL.h.m2 = r4$rate.output
), file.path(od, "ref_convert_rate.csv"), row.names = FALSE)

# ---------- 5. convert_MR ----------
m1 <- convert_MR(0.001, from = "mg/sec",   to = "umol/min",    S = 35, t = 20, quiet = TRUE)
m2 <- convert_MR(0.001, from = "mg/sec/g", to = "umol/min/kg", S = 35, t = 20, quiet = TRUE)
m3 <- convert_MR(0.001, from = "mg/sec/m2", to = "umol/min/cm2", S = 35, t = 20, quiet = TRUE)
write.csv(data.frame(
  abs = m1, mass.spec = m2, area.spec = m3
), file.path(od, "ref_convert_MR.csv"), row.names = FALSE)

cat("reference outputs written to", od, "\n")
