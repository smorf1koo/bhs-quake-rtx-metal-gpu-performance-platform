# GPU Performance / Platform Engineer — тестовое задание «Размытие на GPU: измерить и объяснить»

Bear Head Studio · Quake RTX на Apple Metal

Реализуйте гауссово размытие изображения 4K (σ = 4) на GPU в двух версиях — наивной и одной оптимизированной — и проверьте обе против CPU-эталона. Главное в задании — корректный замер времени на GPU (прогрев, серия замеров, медиана и разброс) и объяснение результата в коротком отчёте.

Задание — в документе [`Тестовое_задание_GPU_Performance_Platform_Engineer.docx`](Тестовое_задание_GPU_Performance_Platform_Engineer.docx).

## Состав пакета

| Путь | Что это |
|---|---|
| `Тестовое_задание_GPU_Performance_Platform_Engineer.docx` | задание: задача, входные данные, что сделать, как сдать |
| `assets/test_4k.png` | тестовое изображение 3840×2160, RGB 8 бит |
| `cpu_reference/` | CPU-эталон гауссова размытия на C++17 (спецификация результата), `ppm_io.h`, CLI `blur_ref`; собирается на любой ОС |
| `starter/` | трек Mac: скелет Metal compute — загрузка PNG, буферы, эталон и сравнение, запись CSV готовы; ядра, двухпроходный dispatch и цикл замера помечены `TODO` |
| `starter_cuda/` | трек без Mac: скелет CUDA (`blur_cuda.cu`) — загрузка PPM, буферы, `cudaEvent`, эталон и сравнение, запись CSV готовы; ядра `naive`, `separable_h` / `separable_v`, второй проход и цикл замера помечены `TODO` |
| `templates/report_template.md` | шаблон `report.md` (1–2 страницы и разделы дополнительной части) |
| `templates/results.csv`, `templates/thermal.csv` | заголовки CSV (`thermal.csv` — только для дополнительного пункта о длительной нагрузке) |
| `tools/image_tools.py` | PNG ↔ PPM, сравнение изображений (Python, только стандартная библиотека) |
| `tools/plot_results.py` | графики по `results.csv` / `thermal.csv` (по желанию) |
| `tools/gen_test_image.py` | генератор тестового изображения; запускать не нужно |

Спецификация эталона кратко: 8 бит на канал, clamp-to-edge, r = ceil(3σ), веса exp(−i²/2σ²), нормированные; горизонтальный, затем вертикальный проход, промежуточный результат во float; округление floor(v + 0.5). Допуск для GPU-версий — max |Δ| ≤ 1 на канал. Оба скелета работают с RGBA8 (альфа = 255, размывается как остальные каналы).

## Как начать

В обоих треках наивный вариант называется `naive`, разделимый — `separable` (ядра `separable_h` + `separable_v`); тайловый вариант назовите по-своему, например `tiled`. Число прогонов по умолчанию — 5 прогревов и 50 замеров (`--warmup`, `--runs`).

### Трек Mac

Нужны macOS 14+, Apple Silicon и Command Line Tools (`xcode-select --install`); полный Xcode нужен только для Instruments и GPU Capture. CMake 3.20+ в Command Line Tools не входит: поставьте его через `brew install cmake` или `pip install cmake`.

```sh
cmake -S starter -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/blur_metal --list
./build/blur_metal --variant naive --csv results.csv
```

Без CMake скелет собирается одной командой из корня пакета (запускать тоже из корня: путь к `kernels.metal` относительный):

```sh
clang++ -std=c++17 -O2 -fobjc-arc -Icpu_reference \
  starter/src/main.mm cpu_reference/gaussian_blur_ref.cpp \
  -framework Metal -framework Foundation \
  -framework ImageIO -framework CoreGraphics -o blur_metal
./blur_metal --variant naive --csv results.csv
```

Сборка Release — это `-O2` / `-DCMAKE_BUILD_TYPE=Release`. Metal API Validation в запуске из терминала выключена, если не задана переменная `MTL_DEBUG_LAYER`; в Xcode её выключают в схеме (Diagnostics).

### Трек без Mac: CUDA

Нужен компьютер с видеокартой NVIDIA и CUDA Toolkit (`nvcc`) или бесплатный Google Colab: «Среда выполнения → Сменить среду выполнения → T4 GPU», затем загрузите архив пакета через панель «Файлы», распакуйте его (`!unzip <архив>.zip`) и выполняйте команды ниже в ячейках с префиксом `!`, перейдя в папку пакета через `%cd`.

```sh
python3 -I tools/image_tools.py to-ppm assets/test_4k.png test_4k.ppm
nvcc -O3 -std=c++17 -Icpu_reference -o blur_cuda \
     starter_cuda/blur_cuda.cu cpu_reference/gaussian_blur_ref.cpp
./blur_cuda --variant naive --csv results.csv
./blur_cuda --variant separable --csv results.csv --output out.ppm
```

Сборка Release для CUDA — `-O3` без `-G` (флаг `-G` выключает оптимизации кода на GPU). В Colab запишите в отчёт модель GPU из `nvidia-smi`: соседи по машине и частоты там не под вашим контролем, поэтому разброс замеров обычно больше, чем на своей машине.

### Эталон и сравнение изображений (любая ОС)

```sh
cmake -S cpu_reference -B build-ref && cmake --build build-ref
python3 -I tools/image_tools.py to-ppm assets/test_4k.png test_4k.ppm
./build-ref/blur_ref test_4k.ppm ref.ppm 4.0
python3 -I tools/image_tools.py compare ref.ppm out.ppm --tolerance 1
```

Без CMake: `clang++ -std=c++17 -O2 -Icpu_reference cpu_reference/blur_ref_main.cpp cpu_reference/gaussian_blur_ref.cpp -o blur_ref` (или `g++` с теми же флагами).
