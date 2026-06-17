const form = document.getElementById('submit-button')
const serverPortrait = document.getElementById('server-portrait')
const clientPortrait = document.getElementById('client-portrait')
const serverPickName = document.getElementById('server-pick-name')
const clientPickName = document.getElementById('client-pick-name')
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
const serverDeckContainer = document.getElementById('server-deck-builder')
const clientDeckContainer = document.getElementById('client-deck-builder')
const charModal = document.getElementById('char-modal')
const charModalGrid = document.getElementById('char-modal-grid')

let charactersList = []
let serverCharId = 0
let clientCharId = 1
let serverDeck = {}
let clientDeck = {}
let serverCardsData = null
let clientCardsData = null
let activePicker = null

async function loadCharacters() {
  const response = await fetch('/api/characters')
  if (!response.ok) { console.error('Falha ao carregar personagens'); return }
  charactersList = await response.json()
  await updateAll()
}

function getCharacterById(id) {
  return charactersList.find(c => c.id === Number(id)) || null
}

function deckTotal(deck) { return Object.values(deck).reduce((sum, n) => sum + n, 0) }

function deckToVisual(deckObj) {
  const slots = []
  for (const [cid, count] of Object.entries(deckObj)) {
    for (let i = 0; i < count; i++) slots.push(Number(cid))
  }
  while (slots.length < 20) slots.push(null)
  return slots
}

function getCardImage(charId, card) {
  return card.image_url || ''
}

async function loadCardsData(charId) {
  const r = await fetch(`/api/character-cards-all/${charId}`)
  if (r.ok) return await r.json()
  return { system: [], skills: {}, spells: {} }
}

async function loadOptimizedAsDeck(charId, deckObj) {
  const r = await fetch(`/api/optimized-deck/${charId}`)
  if (r.ok) {
    const data = await r.json()
    for (const k in deckObj) delete deckObj[k]
    data.deck.forEach(c => { deckObj[c.card_id] = c.count })
  }
}

function renderDeckBuilder(container, charId, deckObj, cardsData) {
  if (!cardsData) return
  container.innerHTML = ''

  const visual = document.createElement('div')
  visual.className = 'deck-visual'
  const slots = deckToVisual(deckObj)
  slots.forEach(cardId => {
    const slot = document.createElement('div')
    slot.className = 'deck-slot'
    if (cardId !== null) {
      const allCards = [...(cardsData.system||[]), ...Object.values(cardsData.skills||{}).flat(), ...Object.values(cardsData.spells||{}).flat()]
      const card = allCards.find(c => c.card_id === cardId)
      if (card) {
        slot.innerHTML = `<img src="${getCardImage(charId, card)}" alt="" onerror="this.style.display='none'" />`
      }
      slot.innerHTML += `<span class="slot-count">${deckObj[cardId] || 0}</span>`
    }
    visual.appendChild(slot)
  })
  container.appendChild(visual)

  const title = document.createElement('div')
  title.className = 'deck-title'
  const total = deckTotal(deckObj)
  title.innerHTML = `<span>${total}/20 cartas</span><strong class="${total===20?'total-ok':''}">${total===20?'Completo':''}</strong>`
  container.appendChild(title)

  renderSystemSection(container, cardsData.system || [], deckObj, charId)
  renderSkillsSection(container, cardsData.skills || {}, deckObj, charId)
  renderSpellsSection(container, cardsData.spells || {}, deckObj, charId)

  const actions = document.createElement('div')
  actions.className = 'deck-actions'
  actions.innerHTML = `<button class="btn-deck btn-optimize">Usar otimizado</button><button class="btn-deck btn-clear">Limpar</button>`
  container.appendChild(actions)

  actions.querySelector('.btn-optimize').addEventListener('click', async () => {
    const btn = actions.querySelector('.btn-optimize')
    btn.disabled = true; btn.textContent = 'Carregando...'
    await loadOptimizedAsDeck(charId, deckObj)
    btn.disabled = false; btn.textContent = 'Usar otimizado'
    renderDeckBuilder(container, charId, deckObj, cardsData)
  })
  actions.querySelector('.btn-clear').addEventListener('click', () => {
    for (const k in deckObj) delete deckObj[k]
    renderDeckBuilder(container, charId, deckObj, cardsData)
  })
}

function updateDeckCounts(container, charId, deckObj, cardsData) {
  const total = deckTotal(deckObj)
  const title = container.querySelector('.deck-title')
  if (title) title.innerHTML = `<span>${total}/20 cartas</span><strong class="${total===20?'total-ok':''}">${total===20?'Completo':''}</strong>`

  const visual = container.querySelector('.deck-visual')
  if (visual) {
    visual.innerHTML = ''
    const slots = deckToVisual(deckObj)
    slots.forEach(cardId => {
      const slot = document.createElement('div')
      slot.className = 'deck-slot'
      if (cardId !== null) {
        const allCards = [...(cardsData.system||[]), ...Object.values(cardsData.skills||{}).flat(), ...Object.values(cardsData.spells||{}).flat()]
        const card = allCards.find(c => c.card_id === cardId)
        if (card) slot.innerHTML = `<img src="${getCardImage(charId, card)}" alt="" onerror="this.style.display='none'" />`
        slot.innerHTML += `<span class="slot-count">${deckObj[cardId] || 0}</span>`
      }
      visual.appendChild(slot)
    })
  }

  container.querySelectorAll('.card-row').forEach(row => {
    const cardId = Number(row.dataset.cardId)
    const count = deckObj[cardId] || 0
    const cntSpan = row.querySelector('.card-row-count')
    if (cntSpan) cntSpan.textContent = count + 'x'
    const minus = row.querySelectorAll('.btn-cnt')[0]
    const plus = row.querySelectorAll('.btn-cnt')[1]
    if (minus) minus.disabled = count <= 0
    if (plus) plus.disabled = total >= 20 || count >= 4
  })
}

function renderSystemSection(container, cards, deckObj, charId) {
  if (!cards.length) return
  const sec = document.createElement('details')
  sec.className = 'card-group'
  sec.innerHTML = `<summary>System (${cards.length})</summary>`
  const list = document.createElement('div')
  list.className = 'card-picks'
  cards.forEach(c => list.appendChild(renderCardRow(c, deckObj, charId, container)))
  sec.appendChild(list)
  container.appendChild(sec)
}

function renderSkillsSection(container, skills, deckObj, charId) {
  const inputs = Object.keys(skills).sort()
  inputs.forEach(inp => {
    const cards = skills[inp]
    const sec = document.createElement('details')
    sec.className = 'card-group'
    sec.innerHTML = `<summary>Skill ${inp} (${cards.length})</summary>`
    const list = document.createElement('div')
    list.className = 'card-picks'
    cards.forEach(c => list.appendChild(renderCardRow(c, deckObj, charId, container)))
    sec.appendChild(list)
    container.appendChild(sec)
  })
}

function renderSpellsSection(container, spells, deckObj, charId) {
  const costs = Object.keys(spells).sort((a,b) => Number(a)-Number(b))
  costs.forEach(cost => {
    const cards = spells[cost]
    const sec = document.createElement('details')
    sec.className = 'card-group'
    sec.innerHTML = `<summary>Spell Cost ${cost} (${cards.length})</summary>`
    const list = document.createElement('div')
    list.className = 'card-picks'
    cards.forEach(c => list.appendChild(renderCardRow(c, deckObj, charId, container)))
    sec.appendChild(list)
    container.appendChild(sec)
  })
}

function renderCardRow(card, deckObj, charId, container) {
  const row = document.createElement('div')
  row.className = 'card-row'
  row.dataset.cardId = card.card_id
  const count = deckObj[card.card_id] || 0
  const total = deckTotal(deckObj)
  const canAdd = total < 20 && count < 4
  const canRemove = count > 0
  const imgSrc = getCardImage(charId, card)

  row.innerHTML = `
    <img class="card-row-icon" src="${imgSrc}" alt="" onerror="this.style.display='none'" />
    <span class="card-row-name">${card.name}</span>
    <span class="card-row-count">${count}x</span>
    <button class="btn-cnt" ${canRemove?'':'disabled'}>-</button>
    <button class="btn-cnt" ${canAdd?'':'disabled'}>+</button>
  `

  const [minus, plus] = row.querySelectorAll('.btn-cnt')
  const cardsData = container === serverDeckContainer ? serverCardsData : clientCardsData

  minus.addEventListener('click', () => {
    const cur = deckObj[card.card_id] || 0
    if (cur <= 0) return
    deckObj[card.card_id] = cur - 1
    if (deckObj[card.card_id] <= 0) delete deckObj[card.card_id]
    updateDeckCounts(container, charId, deckObj, cardsData)
  })
  plus.addEventListener('click', () => {
    const cur = deckObj[card.card_id] || 0
    const tot = deckTotal(deckObj)
    if (tot >= 20 || cur >= 4) return
    deckObj[card.card_id] = cur + 1
    updateDeckCounts(container, charId, deckObj, cardsData)
  })

  return row
}

async function updateChar(charId, isServer) {
  const ch = getCharacterById(charId)
  if (!ch) return
  if (isServer) {
    serverCharId = charId
    serverPortrait.src = `/character-image/${charId}`
    serverPickName.textContent = ch.name
    matchupServerImg.src = `/select-splash/${charId}`
    matchupServerName.textContent = ch.name
  } else {
    clientCharId = charId
    clientPortrait.src = `/character-image/${charId}`
    clientPickName.textContent = ch.name
    matchupClientImg.src = `/select-splash/${charId}`
    matchupClientName.textContent = ch.name
  }
}

async function updateAll() {
  resultCard.hidden = true

  updateChar(serverCharId, true)
  updateChar(clientCharId, false)

  serverCardsData = await loadCardsData(serverCharId)
  clientCardsData = await loadCardsData(clientCharId)

  if (Object.keys(serverDeck).length === 0) {
    await loadOptimizedAsDeck(serverCharId, serverDeck)
  }
  if (Object.keys(clientDeck).length === 0) {
    await loadOptimizedAsDeck(clientCharId, clientDeck)
  }

  renderDeckBuilder(serverDeckContainer, serverCharId, serverDeck, serverCardsData)
  renderDeckBuilder(clientDeckContainer, clientCharId, clientDeck, clientCardsData)
}

function openCharPicker(isServer) {
  activePicker = isServer
  charModalGrid.innerHTML = ''
  charactersList.forEach(ch => {
    const pick = document.createElement('div')
    pick.className = 'char-pick'
    pick.innerHTML = `<img src="/character-image/${ch.id}" alt="${ch.name}" /><div class="char-pick-name">${ch.name}</div>`
    pick.addEventListener('click', () => {
      if (isServer) { serverCharId = ch.id; serverDeck = {} }
      else { clientCharId = ch.id; clientDeck = {} }
      charModal.classList.add('hidden')
      updateAll()
    })
    charModalGrid.appendChild(pick)
  })
  charModal.classList.remove('hidden')
}

document.getElementById('server-pick').addEventListener('click', () => openCharPicker(true))
document.getElementById('client-pick').addEventListener('click', () => openCharPicker(false))
document.querySelector('.char-modal-bg').addEventListener('click', () => charModal.classList.add('hidden'))

function deckToList(deckObj) {
  const lst = []
  for (const [cid, count] of Object.entries(deckObj)) {
    for (let i = 0; i < count; i++) lst.push(Number(cid))
  }
  return lst
}

function getMatchupIndicator(probability) {
  if (probability >= 0.62) return 'Advantage Server'
  if (probability <= 0.38) return 'Advantage Client'
  return 'Neutral'
}

function showResult(result) {
  winnerText.textContent = result.winner === 'server' ? 'Servidor' : 'Cliente'
  serverProbText.textContent = `${(result.server_win_probability * 100).toFixed(1)}%`
  clientProbText.textContent = `${(result.client_win_probability * 100).toFixed(1)}%`
  matchupIndicator.textContent = result.matchup || 'N/A'
  const models = { deck: 'Modelo com deck', trained: 'Modelo treinado', fallback: 'Fallback' }
  modelTypeText.textContent = models[result.model] || result.model
  resultCard.hidden = false
}

form.addEventListener('click', async (event) => {
  event.preventDefault()
  form.disabled = true
  try {
    const payload = {
      server_rank: Number(document.getElementById('server_rank').value),
      client_rank: Number(document.getElementById('client_rank').value),
      server_char: serverCharId,
      client_char: clientCharId,
      server_cards: deckToList(serverDeck),
      client_cards: deckToList(clientDeck),
    }
    const response = await fetch('/api/predict', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) })
    const data = await response.json()
    if (!response.ok) { alert(data.error || 'Erro'); return }
    showResult(data)
  } catch (error) { console.error(error); alert('Erro ao processar previsao.') }
  finally { form.disabled = false }
})

form.disabled = true
loadCharacters().then(() => { form.disabled = false })
