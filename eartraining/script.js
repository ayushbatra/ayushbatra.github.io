
let lastNote = "E_S1";
let totalNotesClicked = 0; // Counter for total notes clicked
let correctNotesClicked = 0; // Counter for correct notes clicked
let already_guessed_wrong = 0;

// const buttons = document.querySelectorAll('.note-button');
const tanpuraAudio = document.getElementById('tanpura');
const volumeSlider = document.getElementById('volume-slider');

const guessButtons = document.querySelectorAll('.guess-button');
const playButtons = document.querySelectorAll('.play-button');


const audioCache = {};
const all_notes = ["E_S1", "E_R_KOMAL", "E_R", "E_G_KOMAL", "E_G", "E_M", "E_M_TEEVRA", "E_P", "E_D_KOMAL", "E_D", "E_N_KOMAL", "E_N", "E_S2"];
// const all_notes = ["E_S1", "E_R",  "E_G", "E_M","E_P",  "E_D", "E_N", "E_S2"];

// Preload audio files into cache
function preloadAudio() {
    all_notes.forEach(note => {
        const audio = new Audio(`data/${note}.wav`);
        audioCache[note] = audio; // Store audio in cache
    });
}
preloadAudio()

// Set initial volume
tanpuraAudio.volume = volumeSlider.value;

// tanpuraAudio.volume = 0.2;


// Update audio volume when the slider value changes
volumeSlider.addEventListener('input', (event) => {
    tanpuraAudio.volume = event.target.value;
});


const totalCountDisplay = document.getElementById('total-count');
const correctCountDisplay = document.getElementById('correct-count');


function resetButtonColors() {
    guessButtons.forEach(button => {
        button.style.backgroundColor = 'white'; // Reset background color to white
    });
}
playButtons.forEach(button => {
    button.addEventListener('click', () => {
        const note = button.dataset.note;
        playSound(note); // Play the sound associated with the button
    });
});


guessButtons.forEach(button => {
    button.addEventListener('click', () => {
        console.log("Guess button pressed")
        const note = button.dataset.note;
        // totalCountDisplay++;
        if (already_guessed_wrong!=1){
        totalNotesClicked++;
        totalCountDisplay.textContent = totalNotesClicked;
        }


        if (note !== lastNote) {
            // Change the color of the button to red
            button.style.backgroundColor = 'red';
            already_guessed_wrong = 1;
            startPitchEngine();
        }
        else{
            // correctCountDisplay++;
            if (already_guessed_wrong!=1){
                correctNotesClicked++;
                correctCountDisplay.textContent = correctNotesClicked;
            }
            button.style.backgroundColor = 'blue';
            already_guessed_wrong = 0;
            resetButtonColors()
            const randomIndex = Math.floor(Math.random() * all_notes.length);
            const randomNote = all_notes[randomIndex];
            lastNote = randomNote
            playSound(lastNote);
            startPitchEngine();
        }
    });
});


document.getElementById('retry-button').addEventListener('click', () => {
    if (lastNote) {
        startPitchEngine();
        playSound(lastNote);
    }    
    if (tanpuraAudio.paused){
        tanpuraAudio.play();

    }
});

// function playSound(note) {
//     const audio = new Audio(`data/${note}.wav`);
//     audio.play();
//     // lastNote = note;  // Store the last note played
// }
// function playSound(note) {
//     const audio = audioCache[note];
//     if (audio) {
//         const newAudio = new Audio(audio.src); // Create a new instance for each play
//         newAudio.play(); // Play the sound

//         // audio.play();
//     } else {
//         console.error(`Audio not cached for note: ${note}`);
//     }
// }
function playSound(note) {
    const audio = audioCache[note];

    if (audio) {
        // If audio is already playing, stop it and reset the time
        if (!audio.paused) {
            audio.pause(); // Stop the current sound
            audio.currentTime = 0; // Reset to the start of the audio
        }
        audio.play(); // Play the sound again
    } else {
        console.error(`Audio not cached for note: ${note}`);
    }
}


function sleep(milliseconds) {
    return new Promise(resolve => setTimeout(resolve, milliseconds));
}


// tanpuraAudio.play();
document.body.addEventListener('click', () => {
    if (tanpuraAudio.paused) {
        tanpuraAudio.play();
    }
});

// ------- PITCH DETECTION (mic) -------
let audioCtxPitch = null;
let analyser = null;
let pitchBuf = null;

async function startPitchEngine() {
    console.log("started pitch engine")
    if (!audioCtxPitch) {
        audioCtxPitch = new AudioContext();
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const src = audioCtxPitch.createMediaStreamSource(stream);

        analyser = audioCtxPitch.createAnalyser();
        analyser.fftSize = 2048;
        pitchBuf = new Float32Array(analyser.fftSize);

        src.connect(analyser);
        detectLoop();
    }
}

function autoCorrelate(buf, sampleRate) {
    let SIZE = buf.length;
    let MAX = SIZE / 2;
    let bestOffset = -1;
    let bestCorrelation = 0;
    let rms = 0;

    for (let i = 0; i < SIZE; i++) rms += buf[i] * buf[i];
    rms = Math.sqrt(rms / SIZE);
    if (rms < 0.01) return null;

    let lastCorrelation = 1;

    for (let offset = 0; offset < MAX; offset++) {
        let correlation = 0;
        for (let i = 0; i < MAX; i++)
            correlation += Math.abs(buf[i] - buf[i + offset]);

        correlation = 1 - correlation / MAX;

        if (correlation > bestCorrelation) {
            bestCorrelation = correlation;
            bestOffset = offset;
        } else if (correlation < lastCorrelation) {
            return sampleRate / bestOffset;
        }

        lastCorrelation = correlation;
    }

    return null;
}

function freqToMidi(freq) {
    return Math.round(69 + 12 * Math.log2(freq / 440));
}
function midiToClass(midi) {
    return (midi % 12 + 12) % 12;
}

function handleDetectedPitch(freq) {
    const midi = freqToMidi(freq);
    const detectedClass = midiToClass(midi);
    const expectedClass = noteClass[lastNote];

    if (detectedClass === expectedClass) {
        console.log("Correct Note (any octave):", freq.toFixed(1), "Hz");
    } else {
        console.log("Wrong note:", freq.toFixed(1), "Hz");
    }
}

function detectLoop() {
    analyser.getFloatTimeDomainData(pitchBuf);
    const pitch = autoCorrelate(pitchBuf, audioCtxPitch.sampleRate);

    if (pitch) handleDetectedPitch(pitch);

    requestAnimationFrame(detectLoop);
}





