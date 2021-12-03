library(data.table)
data <- fread("mcwf_final.csv")
pdf("qq_mcwf.pdf")
qqnorm(data[[1]])
qqline(data[[1]])
dev.off()

pdf("hist_mcwf.pdf")
hist(data[[1]], breaks = 30, main = "", xlab = "")
dev.off()

probs <- fread("gravity_probs.csv")
samples <- sample(probs[, V1], 10000, replace = T, prob = probs[, V2])
pdf("qq_gravity.pdf")
qqnorm(samples)
qqline(samples)
dev.off()

probs <- fread("no_gr_probs.csv")
samples <- sample(probs[, V1], 10000, replace = T, prob = abs(probs[, V2]))

pdf("qq_no_grav.pdf")
qqnorm(samples)
qqline(samples)
dev.off()

probs <- fread("no_gr_probs_b.csv")
samples <- sample(probs[, V1], 10000, replace = T, prob = abs(probs[, V2]))

pdf("qq_no_gravb.pdf")
qqnorm(samples)
qqline(samples)
dev.off()
