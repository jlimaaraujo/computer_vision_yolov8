import matplotlib.pyplot as plt
import os

class Visualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def plot_count_over_time(self, counter, filename="count_over_time.png"):
        """Plot the number of people in the store over time"""
        plt.figure(figsize=(12, 6))
        
        # Convert timestamps to relative time in minutes from start
        if counter.timestamp_history:
            start_time = counter.timestamp_history[0]
            relative_times = [(t - start_time).total_seconds() / 60 for t in counter.timestamp_history]
            
            plt.plot(relative_times, counter.count_history)
            plt.xlabel('Tempo (minutos)')
            plt.ylabel('Número de pessoas')
            plt.title('Número de pessoas ao longo do tempo')
            plt.grid(True)
            
            # Add annotations for max and min
            max_idx = counter.count_history.index(max(counter.count_history))
            min_idx = counter.count_history.index(min(counter.count_history[1:] or [0]))  # Skip initial 0
            
            plt.annotate(f'Máximo: {counter.count_history[max_idx]}', 
                        xy=(relative_times[max_idx], counter.count_history[max_idx]),
                        xytext=(relative_times[max_idx]+0.5, counter.count_history[max_idx]+1),
                        arrowprops=dict(facecolor='red', shrink=0.05))
            
            if counter.count_history[1:]:  # Only annotate min if we have data
                plt.annotate(f'Mínimo: {counter.count_history[min_idx]}', 
                            xy=(relative_times[min_idx], counter.count_history[min_idx]),
                            xytext=(relative_times[min_idx]+0.5, counter.count_history[min_idx]+1),
                            arrowprops=dict(facecolor='blue', shrink=0.05))
            
            plt.savefig(os.path.join(self.output_dir, filename))
            plt.close()
