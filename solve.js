// Sokoban solver for make.fun hallway-and-rooms levels (malik2112 entry).
//
// Idea: the generator builds one hallway with small rooms hanging off it.
// A box that leaves its room into the hallway can never come back, so each
// room is an independent sub-puzzle and the level optimum P* is the sum of
// the room optima. This solver therefore (1) finds the rooms, (2) solves
// each room with a breadth-first search over pushes, which is push-optimal
// by construction, and (3) stitches the room solutions together with plain
// walks through the hallway. Levels that do not split into rooms fall back
// to one global push search. Everything is budgeted in searched states,
// never in seconds, and every tie breaks in a fixed order, so repeated
// runs produce identical output bytes.

const STEP = [
  { dx: 0, dy: -1, walk: 'u', push: 'U' },
  { dx: 0, dy: 1, walk: 'd', push: 'D' },
  { dx: -1, dy: 0, walk: 'l', push: 'L' },
  { dx: 1, dy: 0, walk: 'r', push: 'R' },
];

const ROOM_LIMIT = 400000;
const GLOBAL_LIMIT = 1000000;

/** Parse XSB text into a board: walls, goal cells, box cells, player cell. */
export function parseBoard(text) {
  const lines = String(text).split('\n').filter((line) => line.length > 0);
  const height = lines.length;
  const width = lines.reduce((m, line) => Math.max(m, line.length), 0);
  const wall = new Uint8Array(width * height).fill(1);
  const goals = [];
  const boxes = [];
  let player = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < lines[y].length; x++) {
      const ch = lines[y][x];
      if (ch === '#') continue;
      const cell = y * width + x;
      wall[cell] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(cell);
      if (ch === '$' || ch === '*') boxes.push(cell);
      if (ch === '@' || ch === '+') player = cell;
    }
  }
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (x === 0 || y === 0 || x === width - 1 || y === height - 1) {
        wall[y * width + x] = 1;
      }
    }
  }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { width, height, wall, goals, boxes, player };
}

/**
 * Mark every cell from which a single box could still be pushed onto a
 * goal. Work backwards from the goals: a cell is live when the box can be
 * pushed from it onto a live cell with room for the player behind it.
 * Boxes on dead cells can never finish, so pushes onto them are pruned.
 */
function computeLive(board) {
  const { width, wall, goals } = board;
  const live = new Uint8Array(wall.length);
  const queue = [];
  for (const goal of goals) {
    live[goal] = 1;
    queue.push(goal);
  }
  for (let head = 0; head < queue.length; head++) {
    const cell = queue[head];
    for (const step of STEP) {
      const delta = step.dy * width + step.dx;
      const from = cell - delta;
      const stand = from - delta;
      if (from < 0 || stand < 0 || from >= wall.length || stand >= wall.length) continue;
      if (wall[from] || wall[stand] || live[from]) continue;
      live[from] = 1;
      queue.push(from);
    }
  }
  return live;
}

/** Flood the cells the player can walk to; returns the smallest cell id. */
function reachableLow(board, blocked, start) {
  const { width, wall } = board;
  const seen = new Uint8Array(wall.length);
  const stack = [start];
  seen[start] = 1;
  let low = start;
  while (stack.length > 0) {
    const cell = stack.pop();
    if (cell < low) low = cell;
    const neighbours = [cell - width, cell + width, cell - 1, cell + 1];
    for (const next of neighbours) {
      if (next < 0 || next >= wall.length) continue;
      if (wall[next] || blocked[next] || seen[next]) continue;
      seen[next] = 1;
      stack.push(next);
    }
  }
  return { low, seen };
}

/** Shortest walk string between two cells, or null when unreachable. */
function walkBetween(board, blocked, from, to) {
  const { width, wall } = board;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const letter = new Array(wall.length).fill('');
  const queue = [from];
  prev[from] = from;
  for (let head = 0; head < queue.length; head++) {
    const cell = queue[head];
    if (cell === to) break;
    for (const step of STEP) {
      const next = cell + step.dy * width + step.dx;
      if (next < 0 || next >= wall.length) continue;
      if (wall[next] || blocked[next] || prev[next] !== -1) continue;
      prev[next] = cell;
      letter[next] = step.walk;
      queue.push(next);
    }
  }
  if (prev[to] === -1) return null;
  let path = '';
  for (let cell = to; cell !== from; cell = prev[cell]) path = letter[cell] + path;
  return path;
}

/**
 * True when the box just pushed onto `target` locks a 2x2 block of walls
 * and boxes while some box in the block sits off its goal. Such a block
 * can never move again, so the push is a dead end.
 */
function locksBlock(board, blocked, onGoal, target) {
  const { width, wall } = board;
  const corners = [target, target - 1, target - width, target - width - 1];
  for (const topLeft of corners) {
    if (topLeft < 0) continue;
    const cells = [topLeft, topLeft + 1, topLeft + width, topLeft + width + 1];
    if (cells[3] >= wall.length) continue;
    if (!cells.every((c) => wall[c] || blocked[c])) continue;
    if (cells.some((c) => blocked[c] && !onGoal[c])) return true;
  }
  return false;
}

/**
 * Breadth-first search over pushes for one board. A state is the sorted
 * box cells plus the player's reachable region reduced to its lowest
 * cell, which keeps equivalent positions merged. Because states expand
 * in push order, the first solved state found uses the fewest pushes.
 * Returns { pushes, moves } or null when the state budget runs out.
 */
function searchPushes(board, limit) {
  const { width, wall, goals } = board;
  const size = wall.length;
  const onGoal = new Uint8Array(size);
  for (const goal of goals) onGoal[goal] = 1;
  const live = computeLive(board);
  if (board.boxes.some((box) => !live[box])) return null;
  const solved = (boxes) => boxes.every((box) => onGoal[box] === 1);

  const blocked = new Uint8Array(size);
  const startBoxes = board.boxes.slice().sort((a, b) => a - b);
  const nodes = [{ boxes: startBoxes, player: board.player, parent: -1 }];
  const visited = new Set();
  for (const box of startBoxes) blocked[box] = 1;
  const startLow = reachableLow(board, blocked, board.player).low;
  for (const box of startBoxes) blocked[box] = 0;
  visited.add(startBoxes.join(',') + '|' + startLow);
  let found = solved(startBoxes) ? 0 : -1;

  for (let head = 0; found < 0 && head < nodes.length; head++) {
    const node = nodes[head];
    const boxes = node.boxes;
    for (const box of boxes) blocked[box] = 1;
    const region = reachableLow(board, blocked, node.player).seen;
    for (let i = 0; i < boxes.length && found < 0; i++) {
      const box = boxes[i];
      for (const step of STEP) {
        const delta = step.dy * width + step.dx;
        const stand = box - delta;
        const target = box + delta;
        if (stand < 0 || stand >= size || target < 0 || target >= size) continue;
        if (!region[stand] || wall[target] || blocked[target] || !live[target]) continue;
        blocked[box] = 0;
        blocked[target] = 1;
        const locked = locksBlock(board, blocked, onGoal, target);
        let nextBoxes = null;
        let key = '';
        if (!locked) {
          nextBoxes = boxes.slice();
          nextBoxes[i] = target;
          nextBoxes.sort((a, b) => a - b);
          const low = reachableLow(board, blocked, box).low;
          key = nextBoxes.join(',') + '|' + low;
        }
        blocked[target] = 0;
        blocked[box] = 1;
        if (locked || visited.has(key)) continue;
        if (nodes.length >= limit) {
          for (const b of boxes) blocked[b] = 0;
          return null;
        }
        visited.add(key);
        nodes.push({ boxes: nextBoxes, player: box, parent: head });
        if (solved(nextBoxes)) found = nodes.length - 1;
      }
    }
    for (const box of boxes) blocked[box] = 0;
  }
  if (found < 0) return null;

  const chain = [];
  for (let at = found; at > 0; at = nodes[at].parent) chain.push(at);
  chain.reverse();
  let moves = '';
  let player = board.player;
  let boxes = startBoxes;
  for (const at of chain) {
    const nextBoxes = nodes[at].boxes;
    const nextSet = new Set(nextBoxes);
    const boxSet = new Set(boxes);
    const from = boxes.find((b) => !nextSet.has(b));
    const to = nextBoxes.find((b) => !boxSet.has(b));
    const delta = to - from;
    const step = STEP.find((s) => s.dy * width + s.dx === delta);
    for (const b of boxes) blocked[b] = 1;
    const walk = walkBetween(board, blocked, player, from - delta);
    for (const b of boxes) blocked[b] = 0;
    if (walk === null || !step) return null;
    moves += walk + step.push;
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: chain.length, moves };
}

/** Cell the player occupies after following a move string. */
function playerAfter(width, start, moves) {
  let cell = start;
  for (const ch of moves.toLowerCase()) {
    if (ch === 'u') cell -= width;
    else if (ch === 'd') cell += width;
    else if (ch === 'l') cell -= 1;
    else cell += 1;
  }
  return cell;
}

/**
 * Split the board into rooms. The hallway is the player's row; its dead
 * cells are cut away, and each connected piece of remaining floor is a
 * room. Returns null when the board is not a set of rooms (a box sits in
 * the hallway, or some piece holds a different number of boxes than
 * goals), in which case the caller solves the whole board at once.
 */
function splitRooms(board) {
  const { width, wall, goals, boxes, player } = board;
  const live = computeLive(board);
  const hallwayRow = Math.floor(player / width);
  const cut = new Uint8Array(wall.length);
  for (let cell = hallwayRow * width; cell < (hallwayRow + 1) * width; cell++) {
    if (!wall[cell] && !live[cell]) cut[cell] = 1;
  }
  if (boxes.some((box) => cut[box])) return null;
  const partOf = new Int32Array(wall.length).fill(-1);
  const rooms = [];
  for (let seed = 0; seed < wall.length; seed++) {
    if (wall[seed] || cut[seed] || partOf[seed] >= 0) continue;
    const cells = [seed];
    partOf[seed] = rooms.length;
    for (let i = 0; i < cells.length; i++) {
      for (const step of STEP) {
        const next = cells[i] + step.dy * width + step.dx;
        if (next < 0 || next >= wall.length) continue;
        if (wall[next] || cut[next] || partOf[next] >= 0) continue;
        partOf[next] = rooms.length;
        cells.push(next);
      }
    }
    rooms.push({ boxes: [], goals: [], finished: false });
  }
  for (const box of boxes) rooms[partOf[box]].boxes.push(box);
  for (const goal of goals) {
    if (partOf[goal] < 0) return null;
    rooms[partOf[goal]].goals.push(goal);
  }
  if (rooms.some((room) => room.boxes.length !== room.goals.length)) return null;
  return rooms.filter((room) => room.boxes.length > 0);
}

/**
 * Solve room by room. While one room is being solved, the other rooms'
 * boxes count as walls (their goals once finished), so the searches stay
 * small and independent. The player simply walks on from wherever the
 * previous room left it.
 */
function solveRooms(board) {
  const rooms = splitRooms(board);
  if (rooms === null) return searchPushes(board, GLOBAL_LIMIT);
  let player = board.player;
  let pushes = 0;
  let moves = '';
  for (const room of rooms) {
    const wall = board.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const cell of other.finished ? other.goals : other.boxes) wall[cell] = 1;
    }
    const subBoard = {
      width: board.width,
      height: board.height,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const result = searchPushes(subBoard, ROOM_LIMIT);
    if (result === null) return null;
    pushes += result.pushes;
    moves += result.moves;
    player = playerAfter(board.width, player, result.moves);
    room.finished = true;
  }
  return { pushes, moves };
}

/**
 * Solve one XSB level and return its LURD move string. An empty string
 * means no solution was found within budget; it is still valid format.
 * Never throws.
 */
export function solveLevel(text) {
  try {
    const board = parseBoard(text);
    if (board.player < 0 || board.boxes.length === 0) return '';
    if (board.boxes.length !== board.goals.length) return '';
    const byRooms = solveRooms(board);
    if (byRooms !== null) return byRooms.moves;
    const whole = searchPushes(board, GLOBAL_LIMIT);
    return whole !== null ? whole.moves : '';
  } catch {
    return '';
  }
}

export const solve = solveLevel;
