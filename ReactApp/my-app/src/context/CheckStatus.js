import React,{createContext, useState} from 'react'
export const CheckStatus=createContext();
const CheckStatusProvider=(props)=>{
    const [symList,setsymList]=useState([])
    const setDataValues=(props) =>{
        console.log(props)
        setsymList(props)
    }
    return(
        <CheckStatus.Provider value={{symList,setDataValues}}>
        {props.children}
        </CheckStatus.Provider>
    )
}
export default CheckStatusProvider