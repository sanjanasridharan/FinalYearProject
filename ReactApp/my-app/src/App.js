import React from 'react';
import './App.css';
import { BrowserRouter as Router, Route,Switch} from "react-router-dom"
import LoginPage from './components/LoginPage'
import MainPage from './components/MainPage';
import CheckStatusProvider from './context/CheckStatus'
function App() {
  return (
    <div className="App">
    <CheckStatusProvider>
     <Router> 
          <Switch>
            <Route path="/MainPage" component={MainPage}/>
            <Route path='/' component={LoginPage}/>
            </Switch>
      </Router>
      </CheckStatusProvider>
    </div>
  );
}

export default App;

