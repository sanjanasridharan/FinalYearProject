import React,{useEffect}from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Modal from '@material-ui/core/Modal';
import Button from '@material-ui/core/Button';
import Container from '@material-ui/core/Container';
import {CheckStatus} from '../context/CheckStatus'
import '../image.css'
 
function getModalStyle() {

  const top = 50 ;
  const left = 50 ;
  return {
    top: `${top}%`,
    left: `${left}%`,
    transform: `translate(-${top}%, -${left}%)`,
  };
}
 
const useStyles = makeStyles(theme => ({
  paper: {
    position: 'relative',
    height : 250,
    width: 380,
    border: '2px solid #000',
    boxShadow: theme.shadows[5],
    // backgroundImage: "url(./assests/doctor.jpg)"
  },
  reward: {
    color : "#1A6390",
    marginTop:-5,
    fontSize:22
    // marginLeft:35,

  },
  dis:{
    color : "#1A6390",
    marginLeft:175,
    maxWidth:50,
    marginTop:75
  },
  linkTo:{
    marginTop:1,
    marginLeft:315,
    
   
  },
  rewardbutton:{
    
    // marginTop : 175,
    marginLeft :  210,
    padding:0
  }
}));
 
export default function SimpleModal({props}) {
    const user=React.useContext(CheckStatus)
  const classes = useStyles();
  const [modalStyle] = React.useState(getModalStyle);
  const [open, setOpen] = React.useState(true);
  const handleClose = () => {
    setOpen(false);
  };
  
  return (
    <div>
      <Modal aria-labelledby="simple-modal-title" open={open}>
        <div id='ss' style={modalStyle} className={classes.paper}>
            <h2 id="simple-modal-description" className={classes.reward}><b>Predicted as {user.perdictedDis} </b></h2>
            <h1 className={classes.dis} id="simple-modal-description" ><b>Take Care!!</b></h1>
           
            <Button variant="contained" color="primary" onClick={handleClose} className={classes.rewardbutton}>OK</Button>
            <h5 className={classes.linkTo} style={{cursor:'pointer'}} onClick={(e => window.open("https://www.google.com/search?q="+user.perdictedDis, "_blank"))}>More Info</h5>
        </div>
      </Modal>
    </div>
  );
}