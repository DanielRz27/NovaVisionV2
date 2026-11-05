const imagenInput = document.getElementById("imagenInput");
const btnCamara = document.getElementById("btnCamara");
const btnCapturar = document.getElementById("btnCapturar");
const btnCerrarCamara = document.getElementById("btnCerrarCamara");
const camaraContainer = document.getElementById("camaraContainer");
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const fileName = document.getElementById("fileName");
const preview = document.getElementById("preview");
const btnAnalizar = document.getElementById("btnAnalizar");
const resultadoDiv = document.getElementById("resultado");
const infoModelo = document.getElementById("infoModelo");
const btnMuestra = document.getElementById("btnMuestra");
const btnEvaluar = document.getElementById("btnEvaluar");
const muestraResultado = document.getElementById("muestraResultado");
const evaluacionResultado = document.getElementById("evaluacionResultado");

let imagenSeleccionada = null;
let streamCamara = null;

const EMOJI = { perro: "🐶", gato: "🐱" };

function el(etiqueta, clase, texto) {
    const nodo = document.createElement(etiqueta);
    if (clase) nodo.className = clase;
    if (texto !== undefined) nodo.textContent = texto;
    return nodo;
}

function mostrarImagen(file) {
    if (!file) return;
    imagenSeleccionada = file;
    fileName.textContent = file.name;

    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        preview.style.display = "block";
    };
    reader.readAsDataURL(file);

    btnAnalizar.disabled = false;
    resultadoDiv.style.display = "none";
    resultadoDiv.replaceChildren();
}

function mostrarError(titulo, detalle) {
    resultadoDiv.className = "resultado error";
    resultadoDiv.replaceChildren(
        el("div", "resultado-titulo", "❌ " + titulo),
        el("div", "detalle", detalle)
    );
    resultadoDiv.style.display = "block";
}

function mostrarResultado(d) {
    resultadoDiv.className = "resultado " + d.clase;

    const titulo = el("div", "resultado-titulo");
    titulo.append(EMOJI[d.clase] + " Resultado: ", el("strong", "", d.etiqueta));

    const detalle = el("div", "detalle");
    detalle.append(
        "Valor del modelo (ŷ): ",
        el("strong", "", Number(d.valor_y).toFixed(4)),
        document.createElement("br"),
        "Probabilidad estimada de perro: ",
        el("strong", "", Number(d.probabilidad_perro).toFixed(1) + " %"),
        document.createElement("br"),
        "Confianza en la respuesta: ",
        el("strong", "", Number(d.confianza).toFixed(1) + " %")
    );

    resultadoDiv.replaceChildren(titulo, detalle);
    if (d.incierto) {
        resultadoDiv.append(
            el(
                "div",
                "aviso",
                "Resultado poco seguro. El modelo solo distingue perros de gatos " +
                    "y puede equivocarse con otras imágenes."
            )
        );
    }
    resultadoDiv.style.display = "block";
}

async function pedirJSON(url, opciones) {
    const respuesta = await fetch(url, opciones);
    const datos = await respuesta.json();
    if (!respuesta.ok || datos.error) {
        throw new Error(datos.error || "Ocurrió un error en el servidor.");
    }
    return datos;
}

async function cargarInfo() {
    try {
        const d = await pedirJSON("/info");
        const partes = [`${d.n_caracteristicas} características`, `λ = ${d.lambda}`];
        if (d.exactitud_test != null) {
            partes.push(`exactitud en prueba ${Number(d.exactitud_test).toFixed(1)} %`);
        }
        infoModelo.textContent = "Modelo: " + partes.join(" · ");
    } catch (error) {
        infoModelo.textContent = error.message;
        infoModelo.classList.add("info-error");
    }
}

imagenInput.addEventListener("change", function () {
    if (this.files[0]) mostrarImagen(this.files[0]);
});

btnCamara.addEventListener("click", async () => {
    try {
        streamCamara = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment" },
            audio: false,
        });
        video.srcObject = streamCamara;
        camaraContainer.style.display = "block";
        resultadoDiv.style.display = "none";
    } catch (error) {
        console.error(error);
        alert("No se pudo acceder a la cámara. Verifica los permisos del navegador.");
    }
});

function cerrarCamara() {
    if (streamCamara) {
        streamCamara.getTracks().forEach((t) => t.stop());
        streamCamara = null;
    }
    video.srcObject = null;
    camaraContainer.style.display = "none";
}

btnCapturar.addEventListener("click", () => {
    if (!streamCamara) return;
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (width === 0 || height === 0) {
        alert("La cámara todavía no está lista.");
        return;
    }
    canvas.width = width;
    canvas.height = height;
    canvas.getContext("2d").drawImage(video, 0, 0, width, height);

    canvas.toBlob(
        (blob) => {
            if (!blob) {
                alert("No se pudo capturar la imagen.");
                return;
            }
            mostrarImagen(new File([blob], "captura_novavision.jpg", { type: "image/jpeg" }));
            cerrarCamara();
        },
        "image/jpeg",
        0.95
    );
});

btnCerrarCamara.addEventListener("click", cerrarCamara);

btnAnalizar.addEventListener("click", async () => {
    if (!imagenSeleccionada) {
        alert("Primero selecciona una imagen o toma una fotografía.");
        return;
    }
    btnAnalizar.disabled = true;
    btnAnalizar.textContent = "⏳ Analizando...";
    resultadoDiv.style.display = "none";

    const formData = new FormData();
    formData.append("imagen", imagenSeleccionada);

    try {
        mostrarResultado(await pedirJSON("/predecir", { method: "POST", body: formData }));
    } catch (error) {
        mostrarError("Error", error.message);
    } finally {
        btnAnalizar.disabled = false;
        btnAnalizar.textContent = "🔍 Analizar Imagen";
    }
});

btnMuestra.addEventListener("click", async () => {
    btnMuestra.disabled = true;
    muestraResultado.replaceChildren(el("p", "descripcion", "Analizando imágenes..."));
    try {
        const { imagenes } = await pedirJSON("/muestra?n=8");
        const aciertos = imagenes.filter((i) => i.acierto).length;
        const rejilla = el("div", "rejilla");
        imagenes.forEach((i) => {
            const tarjeta = el("div", "tarjeta " + (i.acierto ? "ok" : "mal"));
            const img = document.createElement("img");
            img.src = i.url;
            img.alt = i.archivo;
            tarjeta.append(
                img,
                el("div", "tarjeta-texto", `${EMOJI[i.clase]} ${i.etiqueta} (${i.confianza.toFixed(0)} %)`),
                el("div", "tarjeta-real", (i.acierto ? "✓ Acierto" : "✗ Era " + i.real))
            );
            rejilla.append(tarjeta);
        });
        muestraResultado.replaceChildren(
            el("p", "resumen", `Aciertos: ${aciertos} de ${imagenes.length}`),
            rejilla
        );
    } catch (error) {
        muestraResultado.replaceChildren(el("p", "info-error", error.message));
    } finally {
        btnMuestra.disabled = false;
    }
});

btnEvaluar.addEventListener("click", async () => {
    btnEvaluar.disabled = true;
    evaluacionResultado.replaceChildren(el("p", "descripcion", "Evaluando..."));
    try {
        const m = await pedirJSON("/evaluar");
        const tabla = el("table", "confusion");
        const filas = [
            ["", "Pred. gato", "Pred. perro"],
            ["Real gato", m.VN, m.FP],
            ["Real perro", m.FN, m.VP],
        ];
        filas.forEach((fila, r) => {
            const tr = document.createElement("tr");
            fila.forEach((valor) => tr.append(el(r === 0 ? "th" : "td", "", String(valor))));
            tabla.append(tr);
        });
        evaluacionResultado.replaceChildren(
            el(
                "p",
                "resumen",
                `Exactitud: ${m.exactitud.toFixed(1)} % sobre ${m.total} imágenes`
            ),
            tabla,
            el(
                "p",
                "descripcion",
                `Precisión ${m.precision.toFixed(1)} % · Sensibilidad ${m.sensibilidad.toFixed(1)} % · ` +
                    `Especificidad ${m.especificidad.toFixed(1)} %`
            )
        );
    } catch (error) {
        evaluacionResultado.replaceChildren(el("p", "info-error", error.message));
    } finally {
        btnEvaluar.disabled = false;
    }
});

cargarInfo();
