
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

        }
    });
});


document.getElementById('retry-button').addEventListener('click', () => {
    if (lastNote) {
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

