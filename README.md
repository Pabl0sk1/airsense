<div align="center">

# ✋ AirSense

**Dibuja en el aire con el dedo, controla la app con gestos y mira tu estado de ánimo en tiempo real — solo con la webcam.**

[![tests](https://github.com/Pabl0sk1/airsense/actions/workflows/tests.yml/badge.svg)](https://github.com/Pabl0sk1/airsense/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20%26%20Face%20Landmarker-0097A7)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![Plataforma](https://img.shields.io/badge/plataforma-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)
[![Licencia: MIT](https://img.shields.io/badge/licencia-MIT-green)](LICENSE)

</div>

---

AirSense convierte tu webcam en una pizarra sin contacto. Sigue tu mano con **MediaPipe Hand Landmarker**,
reconoce 9 gestos para dibujar, borrar, deshacer o guardar, y analiza tu rostro con **Face Landmarker**
para estimar tu emoción, contar parpadeos y sonrisas. Todo corre **en local, en CPU y sin internet**
(una vez descargados los modelos).

## ✨ Características

- 🖌️ **Dibujo en el aire** con trazos suavizados (filtro One Euro + curvas Catmull-Rom): un círculo sale redondo.
- 🤚 **9 gestos** reconocidos y una barra de herramientas que se pulsa apuntando con el dedo.
- 🎨 Paleta de 8 colores, grosor ajustable con la pinza, borrador y deshacer.
- 💾 Exporta el dibujo como **PNG**, **PNG con transparencia** y **captura** con la cámara de fondo.
- 🙂 **Estado de ánimo** en vivo (Feliz, Sorprendido, Triste, Enojado, Neutral) con línea de tiempo de 60 s.
- 📊 **Estadísticas de sesión**: FPS, trazos, cobertura, velocidad del dedo, gestos usados, parpadeos/min…
  y un resumen **JSON** al salir.
- ✂️ **Sin rayas accidentales**: al cerrar la mano, sacarla de la cámara o ir a la barra de botones,
  el final involuntario del trazo se borra solo.
- 🥚 **Easter eggs**: algunos dibujos esconden sorpresas… prueba a dibujar una ☕ taza de café o un ❤️ corazón.
- ⚡ Fluido: el vídeo va a la velocidad de la cámara y la IA corre en un hilo aparte.

## 🚀 Inicio rápido

**Requisitos:** Python 3.10+ y una webcam.

### Windows

Doble clic en **`run.bat`**. La primera vez crea el entorno virtual, instala dependencias,
descarga los modelos (~11 MB) y arranca.

### Linux / macOS

```bash
chmod +x run.sh
./run.sh
```

### Manual

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    ·    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python download_models.py
python app.py
```

### Opciones

| Opción | Descripción |
|---|---|
| `--camera N` | Usa la cámara con índice `N` (por defecto se detecta sola) |
| `--source video.mp4` | Procesa un vídeo en lugar de la cámara |
| `--hands 2` | Sigue las dos manos (más lento) |
| `--no-face` | Desactiva el análisis facial (más FPS) |
| `--no-mirror` | No voltear la imagen horizontalmente |
| `--no-enhance` | Sin realce de nitidez/contraste |
| `--fullscreen` | Arranca en pantalla completa |
| `--profile` | Al salir, imprime el tiempo medio por etapa |
| `--version` | Muestra la versión |

Los argumentos también funcionan con los scripts: `run.bat --no-face` o `./run.sh --hands 2`.

## 🎮 Cómo se usa

### 1. Activa el lápiz

Al arrancar, el índice **no dibuja**: es solo un cursor, para que un dedo suelto no pinte por accidente.
Activa el **Lápiz** apuntando 0,7 s al botón de la barra superior, o pulsa `p`.
Elegir **Borrador** (o hacer el gesto rock) también activa el dibujo, en modo borrador.

### 2. Gestos

| Gesto | Acción |
|---|---|
| ☝️ Solo el índice | **Dibujar** (con el lápiz activo) o mover el **cursor** |
| ✌️ Índice + corazón (V) | **Cursor**: apunta 0,7 s a un botón de la barra para pulsarlo |
| 👌 Pinza (pulgar + índice, resto abiertos) | **Grosor** del pincel según la distancia entre los dedos |
| 🖐️ Palma abierta · 1,2 s | **Limpiar** el lienzo |
| 👍 Pulgar arriba · 1 s | **Guardar** el dibujo en `output/` |
| 🤘 Rock (índice + meñique) · 0,8 s | Activar/desactivar el **borrador** |
| 🤙 Llámame (pulgar + meñique) · 0,8 s | **Deshacer** el último trazo |
| ✊ Puño | **Pausa** (no dibuja) |

Mientras mantienes un gesto aparece un **anillo de progreso** sobre la mano, así sabes cuándo se va a disparar la acción.

### 🥚 Easter eggs

| Dibuja… | Y pasa… |
|---|---|
| ☕ Una taza de café (cuerpo + asa, en uno o varios trazos) | 3 s de lluvia de tazas de café |
| ❤️ Un corazón | 3 s de lluvia de corazones |

Los dibujos se reconocen con el algoritmo [$P](https://depts.washington.edu/acelab/proj/dollar/pdollar.html)
(nubes de puntos: no importa el orden ni el número de trazos), comparándolos con plantillas de tazas y
corazones y con figuras "trampa" (U, rectángulo, círculo, casa…) para no dispararse por error.

### 3. Teclado

| Tecla | Acción | Tecla | Acción |
|---|---|---|---|
| `p` | Lápiz on/off | `1`…`8` | Elegir color |
| `e` | Borrador | `+` / `-` | Grosor |
| `u` | Deshacer | `h` | Ocultar/mostrar interfaz |
| `c` | Limpiar | `a` | Ocultar/mostrar ayuda |
| `s` | Guardar | `m` | Espejo on/off |
| `f` | Pantalla completa | `n` | Realce de imagen on/off |
| `q` / `Esc` | Salir | | |

## 📊 Rostro y estadísticas

- **Emoción**: se calcula a partir de los 52 *blendshapes* faciales de MediaPipe (sonrisa, cejas,
  apertura de boca, ojos…). Se muestra la puntuación de cada emoción, la dominante de la sesión
  y una línea de tiempo de los últimos 60 s.
- **Parpadeos y sonrisas**: detectados por flanco con histéresis, para no contar dos veces el mismo evento.
- **Panel de estadísticas**: FPS de vídeo e IA, manos detectadas, trazos, puntos, cobertura del lienzo,
  tiempo dibujando, recorrido y velocidad del índice, gestos más usados.

Al salir se guarda un resumen en `output/sesion_<fecha>.json` (extracto de ejemplo):

```json
{
  "duracion_s": 1032.4,
  "fps_medio": 29.8,
  "manos": { "recorrido_indice_px": 86466, "gestos_detectados": { "point": 71, "open": 66 } },
  "dibujo": { "trazos": 24, "tiempo_dibujando_s": 172.5, "cobertura_lienzo_pct": 3.1 },
  "rostro": { "emocion_dominante": "Neutral", "parpadeos_por_min": 14.2, "sonrisas": 9 }
}
```

## 🧠 Cómo funciona

```
 Cámara ──► hilo de captura ──► bucle principal (30 fps) ──► lienzo + HUD ──► ventana
                                    │        ▲
                        frame nuevo │        │ último resultado
                                    ▼        │
                              hilo de inferencia
                     Hand Landmarker  +  Face Landmarker (1 de cada N)
```

- **Captura en hilo propio**: siempre se usa el frame más reciente y nunca se procesa el mismo dos veces,
  así el dibujo no va con retraso respecto a la mano.
- **Inferencia asíncrona**: manos y rostro se procesan en segundo plano; el vídeo nunca se congela esperando a la IA.
- **Gestos estables**: un gesto solo se acepta tras repetirse varias inferencias seguidas, y mientras dibujas
  un gesto dudoso no corta el trazo (histéresis). V, palma, puño y pinza lo cortan al instante.
- **Recorte de la cola del trazo**: cada punto guarda su instante. Al terminar un trazo se descartan los
  puntos desde que el gesto empezó a dejar de ser "índice" (menos 0,15 s), los últimos 0,3 s si se perdió
  la mano, y el tramo final que sube hacia la barra si fuiste a pulsar un botón.
- **Puntero suave**: filtro One Euro (quieto cuando estás quieto, sin retraso cuando te mueves rápido).
- **Trazos curvos**: los puntos se unen con splines Catmull-Rom, por eso los trazos salen redondos aunque
  la IA entregue 15-20 posiciones por segundo.
- **HUD eficiente**: el texto se renderiza con Pillow (Segoe UI / Arial / DejaVu) y los paneles se cachean.

## ⚙️ Configuración

Todo lo ajustable está en [`airsense/config.py`](airsense/config.py):

| Parámetro | Para qué sirve |
|---|---|
| `FRAME_W`, `FRAME_H` | Resolución de trabajo (1280×720) |
| `PALETTE`, `BRUSH_*` | Colores y grosores del pincel |
| `HOLD_*`, `DWELL_BUTTON` | Tiempo que hay que mantener cada gesto o apuntar a un botón |
| `GESTURE_STABLE_FRAMES`, `DRAW_STOP_FRAMES` | Estabilidad de gestos y del trazo |
| `STROKE_TAIL_TRIM`, `HAND_LOST_TRIM`, `TOOLBAR_TRIM_MAX` | Cuánto se recorta del final de un trazo |
| `EASTER_EGGS`, `RAIN_DURATION` | Activar los easter eggs y duración de la lluvia |
| `POINTER_MIN_CUTOFF`, `POINTER_BETA` | Suavidad vs. respuesta del puntero |
| `FACE_EVERY_N_FRAMES` | Cada cuántas inferencias se analiza el rostro (ahorra CPU) |
| `ENHANCE_IMAGE`, `HUD_TEXT_REFRESH` | Realce de imagen y refresco del texto de los paneles |

## 📁 Estructura

```
airsense/
├── app.py                 bucle principal, acciones y HUD
├── download_models.py     descarga los modelos de MediaPipe a models/
├── run.bat / run.sh       arranque en un clic (Windows / Linux-macOS)
├── airsense/
│   ├── config.py          resolución, paleta, tiempos de gestos, colores
│   ├── camera.py          detección de cámara y captura en hilo
│   ├── hands.py           seguimiento de manos, gestos, filtro One Euro
│   ├── face.py            emoción, parpadeos y sonrisas
│   ├── canvas.py          lienzo, trazos, borrador, deshacer, guardado
│   ├── stats.py           estadísticas de sesión y exportación JSON
│   ├── shapes.py          reconocimiento de dibujos ($P) para los easter eggs
│   ├── effects.py         lluvia de emojis
│   └── ui.py              barra de herramientas, paneles y texto
└── tests/                 tests unitarios (no necesitan cámara ni modelos)
```

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Cubren la clasificación de gestos (con manos sintéticas), el lienzo (trazos, borrador, deshacer, guardado),
los filtros, la detección de emociones y parpadeos, las estadísticas, el lector de cámara,
el recorte de trazos y el reconocimiento de tazas y corazones.
Se ejecutan automáticamente en GitHub Actions en cada push.

## 🛠️ Solución de problemas

| Problema | Solución |
|---|---|
| *No se encontró ninguna cámara* | Cierra otras apps que usen la cámara (Teams, Zoom…), revisa *Configuración → Privacidad → Cámara* o prueba `--camera 1` |
| *Faltan modelos* | Ejecuta `python download_models.py` (usa `--force` para volver a descargarlos) |
| Va lento | Usa `--no-face`, deja `--hands 1` o sube `FACE_EVERY_N_FRAMES` |
| No detecta bien los gestos | Buena luz, la mano entera dentro del encuadre y a 40-80 cm de la cámara |
| Izquierda/derecha invertidas | Pulsa `m` o usa `--no-mirror` |

> **Nota:** los modelos se cargan en memoria en lugar de por ruta porque MediaPipe falla con rutas
> que contienen acentos (como `Programación`).

## 🙌 Créditos

- [MediaPipe](https://ai.google.dev/edge/mediapipe) — Hand Landmarker y Face Landmarker.
- [OpenCV](https://opencv.org/) — captura y dibujo.
- Vatavu, Anthony y Wobbrock, [*$P Point-Cloud Recognizer*](https://depts.washington.edu/acelab/proj/dollar/pdollar.html) (ICMI 2012) — reconocimiento de dibujos.
- Casiez et al., [*1€ Filter*](https://gery.casiez.net/1euro/) (CHI 2012) — suavizado del puntero.

## 📄 Licencia

Distribuido bajo la licencia [MIT](LICENSE).
