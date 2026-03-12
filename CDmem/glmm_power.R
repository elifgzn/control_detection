library(lme4)
library(simr)

# what hit rate do you expect in each condition?
# e.g. low control: 60% recognition, high control: 70% recognition
# convert to log odds
low_control_prob  <- 0.70
high_control_prob <- 0.78  # conservative - small effect

intercept <- log(low_control_prob / (1 - low_control_prob))
slope <- log(high_control_prob / (1 - high_control_prob)) - intercept

cat("Intercept:", intercept, "\n")
cat("Slope (control effect):", slope, "\n")

# build skeleton data
n_participants <- 60
n_trials <- 60

df <- expand.grid(
  participant = factor(1:n_participants),
  control_level = c(-0.5, 0.5),
  trial = 1:n_trials
)
df$recognized <- rbinom(nrow(df), 1, 0.6)  # placeholder outcome

# fit model with your assumed parameters
model_sim <- makeGlmer(
  recognized ~ control_level + (1 | participant),
  fixef = c(intercept, slope),
  #sd_dprime = 0.75
  #ri_sd_log_odds = sd_dprime * (np.pi / np.sqrt(3))
  #print(ri_sd_log_odds)  # ≈ 1.36
  VarCorr = list(1.36),  # based on Schreiner et al., 2024
  family = binomial,
  data = df
)

# power curve
power_curve <- powerCurve(
  model_sim,
  along = "participant",
  breaks = c(20, 25, 30, 35, 40, 45, 50, 55, 60),
  nsim = 1000
)
plot(power_curve)
print(power_curve)


library(pwr)
# effect size for paired t-test
d_high <- 0.78
d_low <- 0.70
sd_pooled <- 0.75
cohens_d <- (d_high - d_low) / sd_pooled  # ≈ 0.11 - very small

pwr.t.test(d = cohens_d, sig.level = 0.05, power = 0.80, type = "paired")
