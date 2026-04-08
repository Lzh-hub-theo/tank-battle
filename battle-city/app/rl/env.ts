import store from '../utils/store'
import * as actions from '../utils/actions'
import { State } from '../reducers'
import { applyAction } from './controller'

const FIXED_DELTA = 16 // ~60FPS

export class TankEnv {
  reset(): State {
    store.dispatch({ type: actions.A.ResetGame })
    store.dispatch({ type: actions.A.StartGame })
    return this.getState()
  }

  step(action: number) {
    // 1️⃣ 应用AI动作（写在 controller.ts）
    applyAction(action)

    // 2️⃣ 推进一帧
    store.dispatch(actions.tick(FIXED_DELTA))

    // 3️⃣ 获取状态
    const state = this.getState()

    // 4️⃣ 计算 done
    const done = state.game.status === 'gameover'

    // 5️⃣ reward（先简单）
    const reward = 0

    return { state, reward, done }
  }

  getState(): State {
    return store.getState()
  }
}