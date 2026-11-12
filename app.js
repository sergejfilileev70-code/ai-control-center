const $ = (id) => document.getElementById(id);
const message = $("message");
message.addEventListener("input", () => $("counter").textContent = `${message.value.length} / 12000`);
function show(id, value) { $(id).textContent = value || "Нет данных."; }
$("send").addEventListener("click", async () => {
  $("error").textContent = "";
  const password = $("password").value;
  if (!password) { $("error").textContent = "Введите пароль приложения."; return; }
  if (message.value.trim().length < 2) { $("error").textContent = "Введите задачу."; return; }
  const button = $("send"); button.disabled = true; button.textContent = "Агенты работают…";
  $("results").classList.add("hidden");
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {"Content-Type":"application/json", "X-App-Password":password},
      body: JSON.stringify({message:message.value, mode:$("mode").value})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Не удалось выполнить запрос.");
    show("final", data.final); show("gpt", data.openai); show("claude", data.claude);
    $("modeLabel").textContent = $("mode").selectedOptions[0].text.split(" → ")[0];
    $("results").classList.remove("hidden");
    $("results").scrollIntoView({behavior:"smooth",block:"start"});
  } catch (err) { $("error").textContent = err.message; }
  finally { button.disabled = false; button.innerHTML = 'Запустить анализ <span>↗</span>'; }
});
