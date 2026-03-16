import numpy as num
import matplotlib.pyplot as plt


def createAndSaveGraphic (filePath: str, dataMatrixSizes: list, dataMatrixTimes: list) -> None:
    """
    create and save graphic
    """
    plt.figure(figsize=(10,5))
    plt.plot(dataMatrixSizes, dataMatrixTimes, marker = 'o', linestyle = '-', color = 'green')
    plt.title('Graphic of the dependence of the execution time of matrix multiplication')
    plt.xlabel('Matrixs size (N x N)')
    plt.ylabel('Time execution (seconds)')
    plt.axhline(0, color='black', linewidth=0.9, ls='--')
    plt.axvline(0, color='black', linewidth=0.9, ls='--')
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.3)

    plt.savefig(filePath, dpi = 300)
    plt.show()

def main():
    sizes = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
    times = [0, 0.072296, 0.585832, 1.973700, 4.642923, 9.177902, 16.099580, 27.256850, 39.934981]

    filePath = "matrixMultiplicationResult.png"
    createAndSaveGraphic(filePath, sizes, times)

    print(f"[RUN] => Graphic was created and saved into a file: {filePath}")

if __name__ == "__main__" :
    main()