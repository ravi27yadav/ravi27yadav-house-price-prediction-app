document.addEventListener('DOMContentLoaded', () => {
    const chatContainer = document.getElementById('chat-container');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const checkoutModal = document.getElementById('checkout-modal');
    const closeModalBtn = document.getElementById('close-modal');
    
    // Elements to update in modal
    const propTypeEl = document.getElementById('prop-type');
    const propValueEl = document.getElementById('prop-value');
    const depositAmountEl = document.getElementById('deposit-amount');

    let currentStep = 0;
    let houseDetails = { type: '', value: 0, deposit: 0 };

    // Initial Bot Message
    setTimeout(() => {
        addBotMessage("Hello! I am the Locus Paygentic Realtor. Tell me what kind of house you are looking for (e.g. '3 BHK in Mumbai' or 'Villa in California').");
        userInput.disabled = false;
        sendBtn.disabled = false;
        userInput.focus();
    }, 1000);

    function addBotMessage(text, isHTML = false) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message bot-msg';
        
        if (isHTML) {
            msgDiv.innerHTML = text;
            // Attach event listener if button exists
            setTimeout(() => {
                const btn = msgDiv.querySelector('.agent-action-btn');
                if (btn) {
                    btn.addEventListener('click', openCheckout);
                }
            }, 100);
        } else {
            msgDiv.textContent = text;
        }
        
        chatContainer.appendChild(msgDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function addUserMessage(text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message user-msg';
        msgDiv.textContent = text;
        chatContainer.appendChild(msgDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function showTyping() {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message bot-msg typing';
        msgDiv.id = 'typing-indicator';
        msgDiv.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
        chatContainer.appendChild(msgDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function removeTyping() {
        const typing = document.getElementById('typing-indicator');
        if (typing) typing.remove();
    }

    function handleUserInput() {
        const text = userInput.value.trim();
        if (!text) return;

        addUserMessage(text);
        userInput.value = '';
        userInput.disabled = true;
        sendBtn.disabled = true;

        showTyping();

        setTimeout(() => {
            removeTyping();
            processInput(text);
        }, 1500 + Math.random() * 1000); // Simulate network delay
    }

    function processInput(text) {
        if (currentStep === 0) {
            // Predict Price
            const basePrice = Math.floor(Math.random() * 500000) + 200000;
            const formattedPrice = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(basePrice);
            const deposit = basePrice * 0.01; // 1% deposit
            const formattedDeposit = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(deposit);
            
            houseDetails.type = text;
            houseDetails.value = formattedPrice;
            houseDetails.deposit = formattedDeposit;

            addBotMessage(`Based on my ML models, the estimated current market value for a "${text}" is around <strong>${formattedPrice}</strong>.`, true);
            
            setTimeout(() => {
                showTyping();
                setTimeout(() => {
                    removeTyping();
                    addBotMessage(`I have located an off-market property matching your criteria. I can secure this property for you right now with a 1% booking deposit of ${formattedDeposit}.<br><button class="agent-action-btn"><i class="fa-solid fa-lock"></i> Secure via Locus Checkout</button>`, true);
                    userInput.disabled = false;
                    sendBtn.disabled = false;
                }, 2000);
            }, 1000);
            
            currentStep = 1;
        } else {
            addBotMessage("Please click the button above to secure your property, or refresh the page to start a new search.");
            userInput.disabled = false;
            sendBtn.disabled = false;
        }
    }

    function openCheckout() {
        propTypeEl.textContent = houseDetails.type;
        propValueEl.textContent = houseDetails.value;
        depositAmountEl.textContent = houseDetails.deposit;
        checkoutModal.classList.add('active');
    }

    closeModalBtn.addEventListener('click', () => {
        checkoutModal.classList.remove('active');
    });

    // Handle Locus Pay button
    document.querySelector('.locus-pay').addEventListener('click', function() {
        this.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';
        setTimeout(() => {
            this.innerHTML = '<i class="fa-solid fa-check"></i> Payment Successful';
            this.style.background = '#4CAF50';
            this.style.color = 'white';
            setTimeout(() => {
                checkoutModal.classList.remove('active');
                addBotMessage("Payment successful via CheckoutWithLocus! The property has been secured for you. My agents will contact you shortly to finalize the paperwork.");
            }, 2000);
        }, 1500);
    });

    sendBtn.addEventListener('click', handleUserInput);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleUserInput();
    });
});
