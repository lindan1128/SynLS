library(reticulate)
library(ggplot2)

# Define the function
ks_analysis <- function(real_data_path, gen_data_path, real_indices = c(1, 2), gen_indices = c(1, 2)) {
  
  # Load real data
  real_data <- py_run_string(paste0("data = np.load('", real_data_path, "')"))$data
  data1 <- as.data.frame(real_data[, , real_indices[1]])
  data2 <- as.data.frame(real_data[, , real_indices[2]])
  data1_p <- (data1 - min(data1)) / (max(data1) - min(data1))
  data2_p <- (data2 - min(data2)) / (max(data2) - min(data2))
  
  # Load generated data
  gen_data <- py_run_string(paste0("data = np.load('", gen_data_path, "')"))$data
  data_gen1 <- as.data.frame(gen_data[, , gen_indices[1]])
  data_gen2 <- as.data.frame(gen_data[, , gen_indices[2]])
  
  # Helper function to perform KS test and create plot
  perform_ks_test <- function(real_data, gen_data) {
    M <- ncol(real_data)
    statistics <- vector("numeric", M)
    p_values <- vector("numeric", M)
    for (col in seq_len(M)) {
      ks_result <- ks.test(real_data[, col], gen_data[, col])
      statistics[col] <- ks_result$statistic
      p_values[col] <- ks_result$p.value
    }
    ks_data <- data.frame(M = 1:M, Statistics = statistics, P_values = p_values)
    p <- ggplot(ks_data, aes(x = M, y = Statistics, color = P_values < 0.05)) +
      geom_point(size = 3) +
      ylim(0, 0.5) +
      scale_color_manual(values = c("gray", "red3")) +
      labs(x = "M", y = "Statistics") +
      theme_minimal() +
      theme(legend.position = "none")
    return(p)
  }
  
  # Perform tests and generate plots
  plot1 <- perform_ks_test(data1_p, data_gen1)
  plot2 <- perform_ks_test(data2_p, data_gen2)
  
  # Return list of plots
  list(Plot1 = plot1, Plot2 = plot2)
}

# Display plots
plots <- ks_analysis(
  real_data_path = "real.npy",
  gen_data_path = "diffusion.npy"
)
print(plots$Plot1)
print(plots$Plot2)
