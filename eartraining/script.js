
let lastNote = "E_S1";
let totalNotesClicked = 0; // Counter for total notes clicked
let correctNotesClicked = 0; // Counter for correct notes clicked
let already_guessed_wrong = 0;

// const buttons = document.querySelectorAll('.note-button');
const tanpuraAudio = document.getElementById('tanpura');
const volumeSlider = document.getElementById('volume-slider');

const guessButtons = document.querySelectorAll('.guess-button');
const playButtons = document.querySelectorAll('.play-button');


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
            const all_notes = ["E_S1", "E_R", "E_G", "E_M", "E_P", "E_D", "E_N", "E_S2"];
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

function playSound(note) {
    const audio = new Audio(`data/${note}.wav`);
    audio.play();
    // lastNote = note;  // Store the last note played
}

function sleep(milliseconds) {
    return new Promise(resolve => setTimeout(resolve, milliseconds));
}


tanpuraAudio.play();
