import React from 'react';
import Button from '@material-ui/core/Button';
import Avatar from '@material-ui/core/Avatar';
import TextField from '@material-ui/core/TextField';
import LockOutlinedIcon from '@material-ui/icons/LockOutlined';
import Typography from '@material-ui/core/Typography';
import Container from '@material-ui/core/Container';
import { useHistory } from "react-router-dom";
// import {LoginTheme} from '../overrides/Theme'
import MenuAppBar from './MenuAppBar'
export default function LoginPage(){
  const history = useHistory();
  const [username,setUsername] = React.useState('')
  const [password,setPassword] = React.useState('')
  const t="Prediction of Symptom and disease"
  const Submit = (e) => {
    e.preventDefault();
    history.push('/MainPage')
  }

  return (
      <div>
          <MenuAppBar props={t}/>
    <Container style={{marginTop:'50px'}} component="main" maxWidth="xs">
      <div>
        <Avatar className="avatar">
          <LockOutlinedIcon/>
        </Avatar>
        <Typography component="h1" variant="h5">Sign in</Typography>
        <form className="form" noValidate >
        {/* <ThemeProvider theme={LoginTheme}> */}
          <TextField variant="filled"  margin="normal" required fullWidth  label="User Name"  value={username}  autoFocus onChange={e => setUsername(e.target.value)} />
          <TextField variant="filled"  fullWidth name="password" label="Password" type="password" id="password"  value={password} onChange={e => setPassword(e.target.value)} />
          <Button style={{marginTop:'25px'}} type="submit" fullWidth variant="contained" color="primary" onClick={Submit}>Sign In</Button>
        {/* </ThemeProvider> */}
        </form>
      </div>
    </Container>
    </div>
  )
  }