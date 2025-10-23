document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const micButton = document.getElementById('mic-button');

    const synth = window.speechSynthesis; // For Text-to-Speech

    // Initial Bot Message (Dynamically added by JS)
    // This will ensure the message is consistent even if JS takes a moment to load
    addMessage("Hello! I'm OrchidAI, your intelligence assistant for the Royal Orchid Hotel. How can I help you today?", "bot-message");


    // Check for Web Speech API compatibility
    if (!('SpeechRecognition' in window) && !('webkitSpeechRecognition' in window)) {
        micButton.style.display = 'none';
        console.warn("Web Speech API (SpeechRecognition) is not supported by this browser.");
    } else {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        recognition.continuous = false; // Only get one result per recognition
        recognition.lang = 'en-US';
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;

        let isRecording = false;

        micButton.addEventListener('click', () => {
            if (isRecording) {
                recognition.stop();
                micButton.classList.remove('recording');
                isRecording = false;
                console.log("Speech recognition stopped.");
            } else {
                recognition.start();
                micButton.classList.add('recording');
                isRecording = true;
                userInput.value = "Listening for your query..."; // More specific listening prompt
                console.log("Speech recognition started.");
            }
        });

        recognition.onstart = () => {
            userInput.value = "Listening for your query...";
            userInput.disabled = true;
            sendButton.disabled = true;
            micButton.classList.add('recording');
        };

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            userInput.value = transcript;
            console.log("Speech recognized:", transcript);
            sendMessage(transcript); // Send the recognized text
        };

        recognition.onerror = (event) => {
            console.error("Speech recognition error:", event.error);
            userInput.value = "";
            userInput.disabled = false;
            sendButton.disabled = false;
            micButton.classList.remove('recording');
            isRecording = false;
            addMessage("OrchidAI encountered a speech recognition error. Please try typing or speak again.", "bot-message"); // Bot-specific error
        };

        recognition.onend = () => {
            console.log("Speech recognition ended.");
            userInput.disabled = false;
            sendButton.disabled = false;
            micButton.classList.remove('recording');
            isRecording = false;
            if (userInput.value === "Listening for your query...") { // Check for the specific listening prompt
                userInput.value = ""; // Clear if nothing was said
            }
        };
    }

    // Function to add a message to the chat UI
    function addMessage(text, type, animate = false) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', type);

        const messageBubble = document.createElement('div');
        messageBubble.classList.add('message-bubble');
        messageBubble.textContent = text;

        messageDiv.appendChild(messageBubble);
        chatMessages.appendChild(messageDiv);

        if (animate) {
            messageDiv.style.opacity = '0';
            messageDiv.style.transform = 'translateY(20px)';
            setTimeout(() => {
                messageDiv.style.transition = 'opacity 0.3s ease-out, transform 0.3s ease-out';
                messageDiv.style.opacity = '1';
                messageDiv.style.transform = 'translateY(0)';
            }, 50);
        }

        chatMessages.scrollTop = chatMessages.scrollHeight; // Auto-scroll to bottom
        return messageDiv; // Return the message element for potential modification (e.g., typing indicator)
    }

    // Function to add a typing indicator
    function addTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.classList.add('message', 'bot-message', 'typing-indicator');
        typingDiv.id = 'typing-indicator'; // Assign an ID to easily remove it

        const bubble = document.createElement('div');
        bubble.classList.add('message-bubble');
        bubble.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';

        typingDiv.appendChild(bubble);
        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return typingDiv;
    }

    // Function to remove the typing indicator
    function removeTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    // Function to play text as speech
    async function playTextAsSpeech(text) {
        try {
            const response = await fetch(`/tts?text=${encodeURIComponent(text)}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const audioBlob = await response.blob();
            const audioUrl = URL.createObjectURL(audioBlob);
            const audio = new Audio(audioUrl);
            audio.play();
            audio.onended = () => {
                URL.revokeObjectURL(audioUrl); // Clean up the object URL
            };
        } catch (error) {
            console.error("Error playing text as speech:", error);
            // Fallback to client-side TTS if backend fails or is not available
            speakWithClientSideTTS(text);
        }
    }

    // Fallback client-side TTS
    function speakWithClientSideTTS(text) {
        if (synth.speaking) {
            console.error('speechSynthesis.speaking');
            return;
        }
        if (text !== '') {
            const utterThis = new SpeechSynthesisUtterance(text);
            utterThis.onend = function (event) {
                console.log('SpeechSynthesisUtterance.onend');
            }
            utterThis.onerror = function (event) {
                console.error('SpeechSynthesisUtterance.onerror', event);
            }
            // Set voice (optional)
            // const voices = synth.getVoices();
            // utterThis.voice = voices.find(voice => voice.name === 'Google US English');
            synth.speak(utterThis);
        }
    }


    // Main function to send a message
    async function sendMessage(messageText = userInput.value.trim()) {
        if (messageText === "") return;

        addMessage(messageText, 'user-message', true); // Add user message to UI
        userInput.value = ''; // Clear input field

        const typingIndicator = addTypingIndicator(); // Show typing indicator

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: messageText }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            removeTypingIndicator(); // Remove typing indicator
            addMessage(data.answer, 'bot-message', true); // Add bot response to UI
            playTextAsSpeech(data.answer); // Play bot response as speech

        } catch (error) {
            console.error('Error sending message:', error);
            removeTypingIndicator(); // Remove typing indicator even on error
            addMessage("OrchidAI is currently unavailable. Please try again later.", "bot-message", true); // Bot-specific error
        }
    }

    // Event listeners
    sendButton.addEventListener('click', () => sendMessage());
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
});