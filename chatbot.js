const chatbotContainer = document.getElementById("container");
const chatbox = document.getElementById("chatbox");
const chatbody = document.getElementById("chatbody");
const messageTextBox = document.getElementById("userInput");
const sendbutton = document.getElementById("send");
function openChatBot() {
  if (chatbox.style.display) {
    chatbox.style.display = "";
    return;
  }
  chatbox.style.display = "flex";
}
sendbutton.addEventListener("click", sendMsg);
document.addEventListener("keypress", (event) => {
  if (event.key === "Enter" && messageTextBox.value !== "") {
    sendMsg();
    chatbody.scrollTop = chatbody.scrollHeight;
  } else if (
    event.key === "Enter" &&
    messageTextBox.value === "" &&
    chatbox.style.display === "flex"
  ) {
    alert("Please enter a message");
  }
});
function sendMsg() {
  if (messageTextBox.value === "") {
    alert("Please enter a message");
    return;
  }
  let message = document.createElement("div");
  message.classList.add("message");
  let messageText = document.createElement("p");
  messageText.classList.add("text");
  messageText.innerText = messageTextBox.value;
  message.appendChild(messageText);
  chatbody.appendChild(message);
  messageTextBox.value = "";
}
