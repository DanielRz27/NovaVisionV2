// ======================================================
// ELEMENTOS HTML
// ======================================================

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


// ======================================================
// VARIABLES
// ======================================================

let imagenSeleccionada = null;

let streamCamara = null;


// ======================================================
// MOSTRAR UNA IMAGEN
// ======================================================

function mostrarImagen(file) {

    if (!file) {
        return;
    }


    imagenSeleccionada = file;


    // Mostrar nombre

    fileName.textContent = file.name;


    // Crear vista previa

    const reader = new FileReader();


    reader.onload = function(event) {

        preview.src = event.target.result;

        preview.style.display = "block";

    };


    reader.readAsDataURL(file);


    // Activar botón analizar

    btnAnalizar.disabled = false;


    // Ocultar resultado anterior

    resultadoDiv.style.display = "none";

    resultadoDiv.innerHTML = "";
}


// ======================================================
// SELECCIONAR IMAGEN
// ======================================================

imagenInput.addEventListener("change", function() {

    const file = this.files[0];


    if (file) {

        mostrarImagen(file);

    }

});


// ======================================================
// ABRIR CÁMARA
// ======================================================

btnCamara.addEventListener("click", async function() {

    try {

        // Solicitar permiso para utilizar la cámara

        streamCamara = await navigator.mediaDevices.getUserMedia({

            video: {
                facingMode: "environment"
            },

            audio: false

        });


        // Colocar cámara en el video

        video.srcObject = streamCamara;


        // Mostrar cámara

        camaraContainer.style.display = "block";


        // Ocultar resultado anterior

        resultadoDiv.style.display = "none";


    } catch (error) {

        console.error(error);


        alert(
            "No se pudo acceder a la cámara. " +
            "Verifica los permisos del navegador."
        );

    }

});


// ======================================================
// CAPTURAR FOTO
// ======================================================

btnCapturar.addEventListener("click", function() {

    if (!streamCamara) {

        return;

    }


    // Obtener dimensiones del video

    const width = video.videoWidth;

    const height = video.videoHeight;


    if (width === 0 || height === 0) {

        alert("La cámara todavía no está lista.");

        return;

    }


    // Configurar canvas

    canvas.width = width;

    canvas.height = height;


    // Obtener contexto

    const context = canvas.getContext("2d");


    // Dibujar imagen actual de la cámara

    context.drawImage(
        video,
        0,
        0,
        width,
        height
    );


    // Convertir canvas a archivo JPEG

    canvas.toBlob(function(blob) {

        if (!blob) {

            alert("No se pudo capturar la imagen.");

            return;

        }


        // Crear archivo

        const file = new File(

            [blob],

            "captura_novavision.jpg",

            {
                type: "image/jpeg"
            }

        );


        // Usar la misma función que el selector

        mostrarImagen(file);


        // Cerrar cámara

        cerrarCamara();


    }, "image/jpeg", 0.95);

});


// ======================================================
// CERRAR CÁMARA
// ======================================================

function cerrarCamara() {

    if (streamCamara) {

        streamCamara.getTracks().forEach(function(track) {

            track.stop();

        });


        streamCamara = null;

    }


    video.srcObject = null;

    camaraContainer.style.display = "none";

}


// ======================================================
// BOTÓN CERRAR CÁMARA
// ======================================================

btnCerrarCamara.addEventListener("click", function() {

    cerrarCamara();

});


// ======================================================
// ANALIZAR IMAGEN
// ======================================================

btnAnalizar.addEventListener("click", async function() {

    if (!imagenSeleccionada) {

        alert(
            "Primero selecciona una imagen " +
            "o toma una fotografía."
        );

        return;

    }


    // Desactivar botón

    btnAnalizar.disabled = true;

    btnAnalizar.textContent = "⏳ Analizando...";


    // Ocultar resultado anterior

    resultadoDiv.style.display = "none";


    // Crear formulario

    const formData = new FormData();


    formData.append(
        "imagen",
        imagenSeleccionada
    );


    try {

        // Enviar imagen a Flask

        const response = await fetch(

            "/predecir",

            {
                method: "POST",
                body: formData
            }

        );


        // Leer respuesta

        const data = await response.json();


        // ==================================================
        // ERROR
        // ==================================================

        if (!response.ok || data.error) {

            resultadoDiv.className =
                "resultado error";


            resultadoDiv.innerHTML = `

                <div class="resultado-titulo">
                    ❌ Error
                </div>

                <div class="detalle">
                    ${
                        data.error ||
                        "Ocurrió un error al analizar la imagen."
                    }
                </div>

            `;


            resultadoDiv.style.display = "block";


            return;

        }


        // ==================================================
        // RESULTADO
        // ==================================================

        const esPerro =
            data.clase.includes("Perro");


        if (esPerro) {

            resultadoDiv.className =
                "resultado perro";

        } else {

            resultadoDiv.className =
                "resultado gato";

        }


        resultadoDiv.innerHTML = `

            <div class="resultado-titulo">

                ${esPerro ? "🐶" : "🐱"}

                Resultado:

                <strong>
                    ${data.clase}
                </strong>

            </div>


            <div class="detalle">

                Valor del modelo (ŷ):

                <strong>
                    ${Number(data.valor_y).toFixed(4)}
                </strong>

                <br><br>

                Confianza estimada:

                <strong>
                    ${Number(data.confianza).toFixed(1)}%
                </strong>

            </div>

        `;


        resultadoDiv.style.display = "block";


    } catch (error) {

        console.error(error);


        resultadoDiv.className =
            "resultado error";


        resultadoDiv.innerHTML = `

            <div class="resultado-titulo">
                ❌ Error de conexión
            </div>

            <div class="detalle">
                No se pudo conectar con el servidor
                de NovaVision.
            </div>

        `;


        resultadoDiv.style.display = "block";


    } finally {

        // Reactivar botón

        btnAnalizar.disabled = false;

        btnAnalizar.textContent =
            "🔍 Analizar Imagen";

    }

});