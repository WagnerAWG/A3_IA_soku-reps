const form = document.getElementById('submit-button')
const serverChar = document.getElementById('server_char')
const clientChar = document.getElementById('client_char')
const serverImage = document.getElementById('server-image')
const clientImage = document.getElementById('client-image')
const serverName = document.getElementById('server-name')
const clientName = document.getElementById('client-name')
const previewServerSplash = document.getElementById('preview-server-splash')
const previewClientSplash = document.getElementById('preview-client-splash')
const matchupServerImg = document.getElementById('matchup-server-img')
const matchupClientImg = document.getElementById('matchup-client-img')
const matchupServerName = document.getElementById('matchup-server-name')
const matchupClientName = document.getElementById('matchup-client-name')
const resultCard = document.getElementById('result-card')
const winnerText = document.getElementById('winner-text')
const serverProbText = document.getElementById('server-prob')
const clientProbText = document.getElementById('client-prob')
const matchupIndicator = document.getElementById('matchup-indicator')
const modelTypeText = document.getElementById('model-type')
const serverCards = document.getElementById('server-cards')
const clientCards = document.getElementById('client-cards')
const serverOptimizedDeck = document.getElementById('server-optimized-deck')
const clientOptimizedDeck = document.getElementById('client-optimized-deck')

let charactersList = []

async function loadCharacters() {
  const response = await fetch('/api/characters')
  if (!response.ok) {
    console.error('Falha ao carregar personagens')
    alert('Falha ao carregar personagens. Recarregue a página.')
    return
  }

  charactersList = await response.json()
  charactersList.forEach((character) => {
    const optionA = document.createElement('option')
    optionA.value = character.id
    optionA.textContent = character.name
    serverChar.appendChild(optionA)

    const optionB = document.createElement('option')
    optionB.value = character.id
    optionB.textContent = character.name
    clientChar.appendChild(optionB)
  })

  serverChar.value = '0'
  clientChar.value = '1'
  await updatePreviews()
}

function getCharacterById(id) {
  return charactersList.find((character) => character.id === Number(id)) || null
}

async function updateCardGrid(charId, container) {
  if (!charId || !container) {
    return
  }

  const response = await fetch(`/api/character-cards/${charId}`)
  if (!response.ok) {
    container.innerHTML = ''
    return
  }

  const cards = await response.json()
  renderCards(cards, container)
}

function renderCards(cards, container) {
  container.innerHTML = ''
  if (!cards.length) {
    const empty = document.createElement('div')
    empty.className = 'empty-cards'
    empty.textContent = 'Nenhuma carta disponível'
    container.appendChild(empty)
    return
  }

  cards.forEach((card) => {
    const item = document.createElement('div')
    item.className = 'card-icon'
    item.innerHTML = `
      <img src="${card.url}" alt="${card.label}" title="${card.label}" />
      <span>${card.label}</span>
    `
    container.appendChild(item)
  })
}

async function updatePreviews() {
  resultCard.hidden = true

  const server = getCharacterById(serverChar.value)
  const client = getCharacterById(clientChar.value)

  const updateTasks = []

  if (server) {
    serverName.textContent = server.name
    serverImage.src = `/character-image/${server.id}`
    serverImage.alt = server.name
    previewServerSplash.src = `/character-image/${server.id}`
    previewServerSplash.alt = `${server.name} splash`
    document.getElementById('preview-server-name').textContent = server.name
    matchupServerImg.src = `/select-splash/${server.id}`
    matchupServerImg.alt = `${server.name} splash`
    matchupServerName.textContent = server.name
    updateTasks.push(updateCardGrid(server.id, serverCards))
    loadOptimizedDeck(server.id, serverOptimizedDeck).catch(() => {
      renderDeck([], serverOptimizedDeck)
    })
  }

  if (client) {
    clientName.textContent = client.name
    clientImage.src = `/character-image/${client.id}`
    clientImage.alt = client.name
    previewClientSplash.src = `/character-image/${client.id}`
    previewClientSplash.alt = `${client.name} splash`
    document.getElementById('preview-client-name').textContent = client.name
    matchupClientImg.src = `/select-splash/${client.id}`
    matchupClientImg.alt = `${client.name} splash`
    matchupClientName.textContent = client.name
    updateTasks.push(updateCardGrid(client.id, clientCards))
    loadOptimizedDeck(client.id, clientOptimizedDeck).catch(() => {
      renderDeck([], clientOptimizedDeck)
    })
  }

  await Promise.all(updateTasks)
}

function getMatchupIndicator(probability) {
  if (probability >= 0.62) {
    return 'Advantage Server'
  }
  if (probability <= 0.38) {
    return 'Advantage Client'
  }
  return 'Neutral'
}

function renderDeck(deck, container) {
  container.innerHTML = ''
  if (!Array.isArray(deck) || !deck.length) {
    const empty = document.createElement('div')
    empty.className = 'empty-cards'
    empty.textContent = 'Baralho otimizado não disponível.'
    container.appendChild(empty)
    return
  }

  const list = document.createElement('ul')
  list.className = 'deck-list-items'
  deck.forEach((card) => {
    const item = document.createElement('li')
    item.textContent = `${card.count}× ${card.name}`
    list.appendChild(item)
  })
  container.appendChild(list)
}

async function loadOptimizedDeck(charId, container) {
  if (!charId || !container) {
    return
  }

  container.innerHTML = '<p class="deck-loading">Carregando baralho otimizado...</p>'

  const response = await fetch(`/api/optimized-deck/${charId}`)
  if (!response.ok) {
    container.innerHTML = ''
    renderDeck([], container)
    return
  }

  const data = await response.json()
  renderDeck(data.deck || [], container)
}

function showResult(result) {
  winnerText.textContent = result.winner === 'server' ? 'Servidor' : 'Cliente'
  serverProbText.textContent = `${(result.server_win_probability * 100).toFixed(1)}%`
  clientProbText.textContent = `${(result.client_win_probability * 100).toFixed(1)}%`
  matchupIndicator.textContent = getMatchupIndicator(
    result.server_win_probability,
  )
  modelTypeText.textContent =
    result.model === 'trained' ? 'Modelo treinado' : 'Fallback'
  resultCard.hidden = false
}

form.addEventListener('click', async (event) => {
  event.preventDefault()
  form.disabled = true

  try {
    const server = getCharacterById(serverChar.value)
    const client = getCharacterById(clientChar.value)

    if (!server || !client) {
      alert('Selecione personagens válidos.')
      return
    }

    updatePreviews()

    const payload = {
      server_rank: Number(document.getElementById('server_rank').value),
      client_rank: Number(document.getElementById('client_rank').value),
      server_char: Number(serverChar.value),
      client_char: Number(clientChar.value),
    }

    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })

    const data = await response.json()
    if (!response.ok) {
      alert(data.error || 'Erro ao calcular previsão.')
      return
    }

    showResult(data)
  } catch (error) {
    console.error('Erro na previsão:', error)
    alert('Erro ao processar previsão. Tente novamente.')
  } finally {
    form.disabled = false
  }
})

serverChar.addEventListener('change', async () => {
  try {
    await updatePreviews()
  } catch (error) {
    console.error('Erro ao atualizar preview:', error)
  }
})

clientChar.addEventListener('change', async () => {
  try {
    await updatePreviews()
  } catch (error) {
    console.error('Erro ao atualizar preview:', error)
  }
})

form.disabled = true
loadCharacters().then(() => {
  form.disabled = false
})
