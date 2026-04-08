import { RLAction } from './action'

// 👇 全局控制变量（替代键盘）
export let aiPressed: string[] = []
export let aiFire = false

export function applyAction(action: RLAction) {
  aiPressed = []
  aiFire = false

  switch (action) {
    case RLAction.UP:
      aiPressed = ['up']
      break
    case RLAction.DOWN:
      aiPressed = ['down']
      break
    case RLAction.LEFT:
      aiPressed = ['left']
      break
    case RLAction.RIGHT:
      aiPressed = ['right']
      break
    case RLAction.FIRE:
      aiFire = true
      break
    case RLAction.UP_FIRE:
      aiPressed = ['up']
      aiFire = true
      break
  }
}