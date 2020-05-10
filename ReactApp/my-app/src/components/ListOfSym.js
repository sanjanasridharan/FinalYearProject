import React from 'react';
import Checkbox from '@material-ui/core/Checkbox';
import FormGroup from '@material-ui/core/FormGroup';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import FormControl from '@material-ui/core/FormControl';
import FormLabel from '@material-ui/core/FormLabel';
export default function ListOfSym({props}) {
  console.log(props)
    return (
      <FormControl  component="fieldset">
        {/* <FormLabel style={{padding:'20px'}} component="legend">Select the symptoms:</FormLabel> */}
        <FormGroup style={{padding:'20px'}} aria-label="position" row>
         {props.map((sym) =>(
          <FormControlLabel
            value="end"
            control={<Checkbox color="primary" />}
            label={sym}
            labelPlacement="end"
          />))}
          
            
        </FormGroup>
      </FormControl>
    );
  }