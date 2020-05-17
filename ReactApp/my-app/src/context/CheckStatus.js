import React,{createContext, useState} from 'react'
export const CheckStatus=createContext();
const CheckStatusProvider=(props)=>{
    const [symList,setsymList]=useState([])
    const [username,setUser]=useState('');
    const [perdictedDis,setDis]=useState('')
    const storeUsername=(props) =>{
        setUser(props)
    }
    const setDataValues=(props) =>{
        console.log(props)
        setsymList(props)
    }
    const setPredictedDis=(props)=>{
        setDis(props)
    }
    return(
        <CheckStatus.Provider value={{symList,setDataValues,username,storeUsername,setPredictedDis,perdictedDis}}>
        {props.children}
        </CheckStatus.Provider>
    )
}
export default CheckStatusProvider