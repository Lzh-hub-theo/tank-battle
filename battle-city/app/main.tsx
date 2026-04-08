import 'normalize.css'
import React from 'react'
import ReactDOM from 'react-dom'
import { Provider } from 'react-redux'
import App from './App'
import './battle-city.css'
import store from './utils/store'
import { TankEnv } from './rl/env'
import { RLAction } from './rl/action'

ReactDOM.render(
  <Provider store={store}>
    <App />
  </Provider>,
  document.getElementById('container'),
)

if (process.env.NODE_ENV === 'development') {
  const env = new TankEnv()

  setTimeout(() => {
    let state = env.reset()

    for (let i = 0; i < 200; i++) {
      const action = Math.floor(Math.random() * 6)
      const result = env.step(action)

      if (result.done) {
        console.log('Episode finished at step', i)
        break
      }
    }

    console.log('RL test finished')
  }, 1000)
}
