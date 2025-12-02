---
title: "Plantilla de entrada de portafolio"
date: 2025-01-01
---

# Plantilla de entrada de portafolio

Completa esta plantilla para cada entrada del portafolio.

## Contexto
Esta actividad busca comparar el rendimiento de una Red Neuronal Convolucional
(CNN) construida manualmente, frente a un modelo preentrenado para la clasificación
de imágenes VGG16 utilizando Transfer Learning.

## Objetivos
- Crear una CNN desde cero con keras para la clasificación de imágenes
- Utilizar un modelo preentrando para la clasificación de imágenes y utilizar Transfer Learning
- Comparar ambos modelos y determinar las ventajas que posee uno sobre el otro

## Actividades
- Cargar datos
- CNN simple desde cero
- Transfer Learning con timm
- Entrenamiento
- Evaluación y comparación

## Desarrollo
Resumen de lo realizado, decisiones y resultados intermedios.

## Evidencias
- Capturas, enlaces a notebooks/repos, resultados, gráficos

## Reflexión
- Qué aprendiste, qué mejorarías, próximos pasos

## Referencias
- Fuentes consultadas con enlaces relativos cuando corresponda


---

## Guía de formato y ejemplos (MkDocs Material)

Usá estos ejemplos para enriquecer tus entradas. Todos funcionan con la configuración del template.

### Admoniciones

!!! note "Nota"
    Este es un bloque informativo.

!!! tip "Sugerencia"
    Considerá alternativas y justifica decisiones.

!!! warning "Atención"
    Riesgos, limitaciones o supuestos relevantes.

### Detalles colapsables

???+ info "Ver desarrollo paso a paso"
    - Paso 1: preparar datos
    - Paso 2: entrenar modelo
    - Paso 3: evaluar métricas

### Código con resaltado y líneas numeradas

```python hl_lines="2 6" linenums="1"
def train(
    data_path: str,
    epochs: int = 10,
    learning_rate: float = 1e-3,
) -> None:
    print("Entrenando...")
    # TODO: implementar
```

### Listas de tareas (checklist)

- [ ] Preparar datos
- [x] Explorar dataset
- [ ] Entrenar baseline

### Tabla de actividades con tiempos

| Actividad           | Tiempo | Resultado esperado               |
|---------------------|:------:|----------------------------------|
| Revisión bibliográfica |  45m  | Lista de fuentes priorizadas     |
| Implementación      |  90m   | Script ejecutable/documentado    |
| Evaluación          |  60m   | Métricas y análisis de errores   |

### Imágenes con glightbox y atributos

Imagen directa (abre en lightbox):

![Diagrama del flujo](../assets/placeholder.png){ width="420" }

Click para ampliar (lightbox):

[![Vista previa](../assets/placeholder.png){ width="280" }](../assets/placeholder.png)

### Enlaces internos y relativos

Consultá también: [Acerca de mí](../acerca.md) y [Recursos](../recursos.md).

### Notas al pie y citas

Texto con una afirmación que requiere aclaración[^nota].

[^nota]: Esta es una nota al pie con detalles adicionales y referencias.

### Emojis y énfasis

Resultados destacados :rocket: :sparkles: y conceptos `clave`.
