const chatbotContainer = document.getElementById("container");
const chatbox = document.getElementById("chatbox");
const chatbody = document.getElementById("chatbody");
const messageTextBox = document.getElementById("userInput");
const sendbutton = document.getElementById("send");
const questions = [
  "Please enter the name of the author whose book you would like to read.",
  "Please enter the name of the publisher whose book you would like to read.",
  "Please enter the partial title of the book you're interested in.",
  "If you'd like, you can select the option to get the top 10 books with the maximum rating.",
  "Alternatively, you can choose to get the book ID based on the title of the book.",
];

function openChatBot() {
  continueConversation("push_questions", null);
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
function chatBotSendList(data) {
  if (Array.isArray(data) === false) {
    let message = document.createElement("div");
    message.classList.add("message");
    message.classList.add("isbot");
    let messageText = document.createElement("p");
    messageText.classList.add("text");
    messageText.innerText = data;
    message.appendChild(messageText);
    chatbody.appendChild(message);
    chatbody.scrollTop = chatbody.scrollHeight;
    return;
  }
  let orderedList = document.createElement("ol");
  orderedList.classList.add("message");
  orderedList.classList.add("isbot");
  data.map((item) => {
    let listItem = document.createElement("li");
    listItem.innerText = item;
    listItem.style.width = "90%";
    orderedList.appendChild(listItem);
  });
  chatbody.appendChild(orderedList);
  chatbody.scrollTop = chatbody.scrollHeight;
}

async function continueConversation(continue_id, query) {
  switch (continue_id) {
    case "push_questions": {
      chatBotSendList(questions);
      break;
    }
    default:
      await fetch(
        `http://127.0.0.1:5000/recommendations/${continue_id}/${query}`
      )
        .then((response) => {
          return response.json();
        })
        .then((data) => {
          chatBotSendList(data);
        });
  }
}

function sendMsg() {
  if (messageTextBox.value === "") {
    alert("Please enter a message");
    return;
  }
  let message = document.createElement("div");
  message.classList.add("message");
  message.classList.add("isme");
  let messageText = document.createElement("p");
  messageText.classList.add("text");
  messageText.innerText = messageTextBox.value;
  message.appendChild(messageText);
  chatbody.appendChild(message);
  const messageValue = messageTextBox.value;
  const splitted = messageValue.split(",");
  if (splitted.length == 2) {
    continueConversation(splitted[0].trim(), splitted[1].trim());
    messageTextBox.value = "";
    return;
  }
  messageTextBox.value = "";
}
