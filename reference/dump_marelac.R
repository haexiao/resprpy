# Dump marelac internals needed by respR
suppressPackageStartupMessages(library(marelac))
mout <- "X:/Rtools/resprpy/reference/r_src/marelac"
dir.create(mout, recursive = TRUE, showWarnings = FALSE)
ns <- asNamespace("marelac")
mfs <- c("gas_solubility", "sw_gibbs", "rho", "rhoH2O_Chen", "checkVecLength")
for (f in mfs) {
  if (exists(f, envir = ns, inherits = FALSE)) {
    fn <- get(f, envir = ns)
    txt <- deparse(fn, control = "all")
    writeLines(txt, file.path(mout, paste0(f, ".R")))
    cat("dumped", f, length(txt), "lines\n")
  } else cat("MISSING", f, "\n")
}
# data objects
d <- get(".marelac", envir = ns)
nm <- names(d)
cat("--- .marelac members:", paste(nm, collapse=", "), "\n")
for (m in c("Waals.a", "Waals.b", "atmComp")) {
  if (!is.null(d[[m]])) {
    writeLines(deparse(d[[m]], control = "all"), file.path(mout, paste0("ml_", m, ".R")))
    cat("dumped ml_", m, "\n", sep="")
  }
}
aw <- marelac::atomicweight
writeLines(deparse(aw, control = "all"), file.path(mout, "ml_atomicweight.R"))
cat("dumped ml_atomicweight, dim:", dim(aw), "\n")
# O2 constants
cat("Waals.a O2:", d$Waals.a["O2"], " Waals.b O2:", d$Waals.b["O2"], "\n")
cat("atmComp O2:", d$atmComp["O2"], "\n")
