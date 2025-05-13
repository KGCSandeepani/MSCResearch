import matplotlib.pyplot as plt
import os

class SummaryPlotter:
    def __init__(self, output_dir="static"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_pie_chart(self, buggy_count, not_buggy_count):
        labels = ["Buggy", "Not Buggy"]
        sizes = [buggy_count, not_buggy_count]
        colors = ["#66cc66", "#df944d"]

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.axis("equal")

        pie_chart_path = os.path.join(self.output_dir, "buggy_pie_chart.png")
        fig.savefig(pie_chart_path)
        plt.close(fig)

        return pie_chart_path



