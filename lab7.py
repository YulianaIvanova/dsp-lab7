from typing import Final

import numpy as np
from scipy.signal import deconvolve, convolve
from scipy.fft import fft
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import pylab

MORSE_CODES: Final[dict[str, str]] = {
    "a": ".-",
    "b": "-...",
    "c": "-.-.",
    "d": "-..",
    "e": ".",
    "f": "..-.",
    "g": "--.",
    "h": "....",
    "i": "..",
    "j": ".---",
    "k": "-.-",
    "l": ".-..",
    "m": "--",
    "n": "-.",
    "o": "---",
    "p": ".--.",
    "q": "--.-",
    "r": ".-.",
    "s": "...",
    "t": "-",
    "u": "..-",
    "v": "...-",
    "w": ".--",
    "x": "-..-",
    "y": "-.--",
    "z": "--..",
}

N = 51

def morse_encode(message: str, unit_length: int) -> np.ndarray:
    dot = np.ones(unit_length)
    dash = np.ones(3 * unit_length)
    letter_intraspace = np.zeros(unit_length)
    letter_interspace = np.zeros(3 * unit_length)
    word_interspace = np.zeros(7 * unit_length)

    encoded = np.zeros((0))
    prev_symbol_is_letter = False

    for symbol in message:
        if symbol in MORSE_CODES:
            if prev_symbol_is_letter:
                encoded = np.concatenate((encoded, letter_interspace))
            is_first = True
            for code in MORSE_CODES[symbol]:
                if is_first:
                    is_first = False
                else:
                    encoded = np.concatenate((encoded, letter_intraspace))
                if code == ".":
                    chunk = dot
                elif code == "-":
                    chunk = dash
                else:
                    msg = f'Invalid morse code "{MORSE_CODES[symbol]}" for "{symbol}".'
                    raise Exception(msg)
                encoded = np.concatenate((encoded, chunk))
            prev_symbol_is_letter = True
        else:
            encoded = np.concatenate((encoded, word_interspace))
            prev_symbol_is_letter = False

    return encoded


def build_lowpass(w: float, n: int) -> np.ndarray:
    # TODO 2.3 Рассчитать ИХ для ФР НЧ КИХ-фильтра с частотой среза w [рад/отсчёт] и размером n [отсчётов]
    ih_befor_zero = np.sin(w*np.arange(-n//2, 0)) / (np.arange(-n//2, 0) * np.pi)
    ih_after_zero = np.sin(w * np.arange(1, n//2 + 1)) / (np.arange(1, n//2 + 1) * np.pi)
    filter_ih = np.concatenate([ih_befor_zero, [w/np.pi], ih_after_zero])

    plt.plot(filter_ih)
    plt.title("2.3 ИХ для ФР НИЧ")
    plt.show()


    return filter_ih


class LowpassReconstruction:
    def __init__(self, recovered: np.ndarray, m : int) -> None:
        self.__recovered = recovered
        self.__m = m

    @property
    def recovered(self) -> np.ndarray:
        return self.__recovered

    @property
    def m(self) -> int:
        return self.__m


def lowpass_reconstruct(y: np.ndarray, h: np.ndarray) -> LowpassReconstruction:
    # TODO 2.1 Развернуть y и h, чтобы получить оценку x
    x_1, remainder = deconvolve(y, h)


    # TODO 2.2 Определить размер одной точки Морзе в отсчётах и соответствующую частоту среза w
    spectrum = fft(x_1 - np.mean(x_1))
    index_max = np.argmax(spectrum[:int(spectrum.shape[0] / 2)])
    print(f'Index max: {index_max}')


    plt.plot(np.abs(spectrum))
    plt.plot(int(index_max), np.abs(spectrum[index_max]), "x")
    plt.title("2.2 Спектр x_1")
    plt.show()


    w0 = 2 * np.pi * index_max / x_1.shape[0]
    print(f'w0 = {w0}')
    M = int(np.round(x_1.shape[0] / (2 * index_max)))
    print(f"Размер одной точки Морзе M: {M}")

    # H_B = np.zeros(x_1.shape[0])
    # for i in range(x_1.shape[0]):
    #     w = 2 * np.pi * i / x_1.shape[0]
    #     if 0 < w < w0:
    #         H_B[i] = 1
    #     else:
    #         H_B[i] = 0
    # #plt.plot(H_B)
    # #plt.show()
    # # for i in range(H_B.shape[0]):
    # #     if H_B[i] == 1:
    # #         print(f'H_b {H_B[i]} index {i}')
    #
    # h_b = np.zeros(H_B.shape[0], dtype="complex")
    # for i in range(H_B.shape[0]):
    #     h_b[i] = H_B[index_max] * np.exp(complex(0,1)*w0*i)
    #
    # print(h_b)
    # plt.plot(h_b)
    # plt.show()


    # TODO 2.4 Построить и применить ФР НЧ КИХ-фильтр
    h_recovery = build_lowpass(w0, N)
    x_recovery = convolve(x_1, h_recovery)

    plt.plot(x_recovery)
    plt.title("2.4 Восстановленный сигнал")
    plt.show()
    del y, h
    return LowpassReconstruction(x_recovery, M)


class SuboptimalReconstruction:
    def __init__(self, recovered: np.ndarray) -> None:
        self.__recovered = recovered
        # Добавить допданные при необходимости

    @property
    def recovered(self) -> np.ndarray:
        return self.__recovered


def suboptimal_reconstruct(
    y: np.ndarray, h: np.ndarray, v: np.ndarray
) -> SuboptimalReconstruction:
    # TODO 4.1 Оценить r_y, r_v

    r_y = np.var(y) * acf(y, nlags=100)
    r_y = np.concatenate((np.flip(r_y), r_y[1:]))
    plt.plot(r_y)
    plt.title("R_y")
    plt.show()


    r_v = np.var(v) * acf(v, nlags=100)
    r_v = np.concatenate((np.flip(r_v), r_v[1:]))
    plt.plot(r_v)
    plt.title("R_v")
    plt.show()

    # TODO 4.2 Оценить r_xy, r_x
    r_xy, _ = deconvolve(np.flip(r_y - r_v), h)
    #r_xy = np.flip(r_xy)
    plt.plot(r_xy)
    plt.title("R_xy")
    plt.show()


    r_x, _ = deconvolve(np.flip(r_xy), h)
    plt.plot(r_x)
    plt.title("R_x")
    plt.show()



    # TODO 4.3 Рассчитать (уравнение Винера-Хопфа) фильтр и применить фильтр
    #vector_r_y * h_recovery = matrix_r_xy

    #center_r_xy = r_xy.shape[0] // 2
    center_r_y = r_y.shape[0] // 2
    #print(f'center_r_xy {center_r_xy} center_r_y {center_r_y}')
    D = np.arange(-N//2, N//2)


    matrix_r_y = np.zeros((N, N))
    for i, m in enumerate(D):
        for j, k in enumerate(D):
            matrix_r_y[i, j] = r_y[center_r_y + k - m]

    vector_r_xy = np.zeros(len(D))
    for i, m in enumerate(D):
        vector_r_xy[i] = r_xy[len(r_y - r_v) // 2 - m]

    vector_r_xy = np.flip(vector_r_xy)

    #vector_r_xy = np.flip(r_xy[center_r_xy - N // 2 : center_r_xy + N // 2 + 1])

    h_recovery = np.linalg.solve(matrix_r_y, vector_r_xy)

    plt.plot(h_recovery)
    plt.title("4.3 фильтр Винера-Хопфа")
    plt.show()

    x_recovery = convolve(y, h_recovery)


    # TODO 4.4 По r_x[0] и ИХ фильтра рассчитать оценку погрешности восстановления
    err = r_x[len(r_x)//2] - np.dot(h_recovery, vector_r_xy)
    print(f"Погрешность восстановления: {err}")
    del y, v, h
    return SuboptimalReconstruction(x_recovery)


def main() -> None:
    # TODO 1 Загрузить данные и оценить h[n]
    data = np.load("33.npy")
    y = np.ravel(data[0, :])
    v = np.ravel(data[1, :])
    h = data[2:, :]

    h = np.mean(h, axis=0)
    h[np.abs(h) < 0.06] = 0
    h = np.trim_zeros(h, "b")

    print(h.shape)
    for i in range(len(h)):
        if h[i] != 0.0:
            print(f"{h[i]}  {i}")


    # TODO 3.1 Восстановить сигнал с помощью lowpass_reconstruct и вручную/автоматически декодировать сообщение
    x_recovery_lowpass = lowpass_reconstruct(y, h)
    M = x_recovery_lowpass.m

    x_rec = x_recovery_lowpass.recovered
    x_rec[x_rec < 0.5] = 0
    x_rec[x_rec > 0.5] = 1

    plt.plot(x_rec)
    plt.title("3.1 После пороговой обработки")
    plt.show()

    # TODO 3.2 С помощью morse_encode сформировать идеальный полезный сигнал и рассчитать MSE
    message = "if you want peace prepare for war"
    encoded_message = morse_encode(message, M)

    print(f'\nlen encoded_message: {encoded_message.shape[0]}')
    print(f'len x_recovery: {x_recovery_lowpass.recovered.shape[0]}')

    shift = N // 2
    n = len(encoded_message)
    mse = np.sum((encoded_message - x_recovery_lowpass.recovered[shift: shift + n])**2) / n
    print(f"\nMSE ФР НИЧ : {mse}")


    # TODO 5 Восстановить сигнал с помощью suboptimal_reconstruct
    x_recovery_subopt = suboptimal_reconstruct(y, h, v)
    plt.plot(x_recovery_subopt.recovered)
    plt.title("Восстановленный сигнал Винера")
    plt.show()


    mse = np.mean((encoded_message - x_recovery_subopt.recovered[shift: shift + n])**2)
    print(f"\nMSE ФР Винера : {mse}")

if __name__ == "__main__":
    main()
