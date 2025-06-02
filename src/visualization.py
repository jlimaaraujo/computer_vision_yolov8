import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker
import numpy as np
import cv2
class Visualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def plot_count_over_time(self, counter, filename):
        """Plot the number of people in the store over time"""
        plt.figure(figsize=(12, 6))
        
        # Convert timestamps to relative time in seconds from start
        if counter.timestamp_history:
            start_time = counter.timestamp_history[0]
            relative_times = [(t - start_time).total_seconds() for t in counter.timestamp_history]
            relative_times_min = [t / 60 for t in relative_times]

            plt.step(relative_times_min, counter.count_history, where='post')  # Use step plot para contagem discreta
            plt.xlabel('Tempo (minutos)')
            plt.ylabel('Número de pessoas')
            plt.title('Número de pessoas ao longo do tempo')
            plt.grid(True)

            # Eixo Y apenas com inteiros
            plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

            # Eixo X com ticks de acordo com a duração real
            plt.xlim(0, max(relative_times_min))
            plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

            # Add annotations for max and min
            max_idx = counter.count_history.index(max(counter.count_history))
            min_idx = counter.count_history.index(min(counter.count_history[1:] or [0]))  # Skip initial 0

            plt.annotate(f'Máximo: {counter.count_history[max_idx]}', 
                        xy=(relative_times_min[max_idx], counter.count_history[max_idx]),
                        xytext=(relative_times_min[max_idx]+0.1, counter.count_history[max_idx]+1),
                        arrowprops=dict(facecolor='red', shrink=0.05))

            if counter.count_history[1:]:  # Only annotate min if we have data
                plt.annotate(f'Mínimo: {counter.count_history[min_idx]}', 
                            xy=(relative_times_min[min_idx], counter.count_history[min_idx]),
                            xytext=(relative_times_min[min_idx]+0.1, counter.count_history[min_idx]+1),
                            arrowprops=dict(facecolor='blue', shrink=0.05))

            plt.savefig(os.path.join(self.output_dir, filename))
            plt.close()
    
    def plot_stay_duration_histogram(self, counter, filename):
        """Plot histogram of stay durations"""
        durations = []
        for person in counter.people_data.values():
            if person['entered'] and person['exited']:
                frames_spent = person['last_seen'] - person['first_seen']
                time_spent = frames_spent / counter.fps / 60  # em minutos
                durations.append(time_spent)
        
        plt.figure(figsize=(10, 6))
        plt.hist(durations, bins=10, color='skyblue', edgecolor='black')
        plt.xlabel('Tempo de permanência (minutos)')
        plt.ylabel('Número de pessoas')
        plt.title('Distribuição dos tempos de permanência')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        
    def plot_cumulative_entries_exits(self, counter, filename):
        """Plot cumulative entries and exits over time"""
        entries = 0
        exits = 0
        cum_entries = []
        cum_exits = []

        for i in range(len(counter.timestamp_history)):
            if i == 0:
                cum_entries.append(0)
                cum_exits.append(0)
            else:
                # A contagem só muda se houver update nesse frame, por isso é seguro usar os totais globais
                cum_entries.append(counter.entries)
                cum_exits.append(counter.exits)

        relative_times = [(t - counter.timestamp_history[0]).total_seconds() / 60 for t in counter.timestamp_history]

        plt.figure(figsize=(12, 6))
        plt.plot(relative_times, cum_entries, label='Entradas acumuladas', color='green')
        plt.plot(relative_times, cum_exits, label='Saídas acumuladas', color='red')
        plt.xlabel('Tempo (minutos)')
        plt.ylabel('Número de pessoas')
        plt.title('Evolução acumulada de entradas e saídas')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        
    def plot_heatmap(self, counter, filename):
        """Plot heatmap of people trajectories"""
        width = max(counter.entry_zone_pixels[2], counter.exit_zone_pixels[2])
        height = max(counter.entry_zone_pixels[3], counter.exit_zone_pixels[3])

        if width == 0 or height == 0:
            print("[ERRO] Dimensões inválidas para o heatmap. Zonas de entrada/saída podem estar mal definidas.")
            return

        heatmap = np.zeros((height, width), dtype=np.float32)

        for person in counter.people_data.values():
            for bbox in person.get('positions', []):
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                if 0 <= center_x < width and 0 <= center_y < height:
                    heatmap[center_y, center_x] += 1

        # Verificar se o heatmap tem valores diferentes de zero
        if np.count_nonzero(heatmap) == 0:
            print("[INFO] Nenhuma trajetória registada. Heatmap não será gerado.")
            return

        heatmap_blur = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=15, sigmaY=15)

        plt.figure(figsize=(12, 6))
        plt.imshow(heatmap_blur, cmap='hot', interpolation='nearest', origin='upper')
        plt.colorbar(label='Frequência de passagem')
        plt.title('Heatmap das trajetórias das pessoas')
        plt.xlabel('Largura do frame (pixels)')
        plt.ylabel('Altura do frame (pixels)')
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
