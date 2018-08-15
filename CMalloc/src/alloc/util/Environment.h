/* 
 *  Copyright (c) 2013, Faculty of Informatics, Masaryk University
 *  All rights reserved.
 *  
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of the <organization> nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *  
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *  Authors:
 *  Tomas Kopal, 1996
 *  Vilem Otte <vilem.otte@post.cz>
 *
 */

#ifndef __ENVIRONMENT_H__
#define __ENVIRONMENT_H__

#include <iostream>
#include <string>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <string.h>

using namespace std;

char *GetPath(const char *s);

/** 
  * Enumeration type used to distinguish the type of env-var
  * @name OptionType
  **/
enum OptionType {
  optInt,			// Integer env-var
  optFloat,			// Floating point env-var
  optBool,			// Boolean env-var
  optString			// String env-var
};

/**
  * Option class abstracts options in environment class
  * @name Option
  **/
class Option
{
public:
    //////////////////////////
    /** @section Variables **/

    OptionType type;        //< Option type
    char* name;             //< Option name
    char* value;            //< Option value
    char* abbrev;           //< Option Abbreviation
    char* defaultValue;     //< Option default value

    char* description;      //< Option description
    void* pointer;          //< TODO: Whats this for? (search in Minimax)
    
    /////////////////////////////
    /** @section Constructors **/

    /**
      * Default constructor
      * @name Option
      * @param None
      * @return None
      **/
    Option()
    {
        name = NULL;
        value = NULL;
        abbrev = NULL;
        defaultValue = NULL;
        description = NULL;
        pointer = NULL;
    }
    
    ////////////////////////////
    /** @section Destructors **/

    /**
      * Default desctructor, virtual, frees allocated memory
      * @name ~Option
      * @param None
      * @return None
      **/
    virtual ~Option()
    {
        if(name) delete [] name;
        if(value) delete [] value;
        if(abbrev) delete [] abbrev;
        if(defaultValue) delete [] defaultValue;
        if(description) delete [] description;
    }
    
    ////////////////////////
    /** @section Methods **/

    /**
      * Friend method, overloads operator '<<' to ostream (e.g. text output to stream), I'd say 
      * that this is suited mainly for debug purposes.
      * @param ostream& - Output stream where we want to apped text info about option
      * @param const COption& - Specifies option we want to output
      * @return ostream& - Returns ostream passed in
      **/
    friend ostream& operator<<(ostream& s, const Option& o)
    {
        // Ouptut name, abbrevation, value, default value and description
        // Ideally resulting in string:
        // > option_small_x  opt_x   12.45   [1.0] 
        // > Small x option
        s << o.name << "\t";

        if(o.abbrev)
            s << o.abbrev;
        else
            s << "no abbrev";
        s << "\t";

        if(o.value)
            s << o.value;
        else
            s << "not set";
        s << "\t";

        if(o.defaultValue)
            s << "[" << o.defaultValue << "]";
        else
            s << "[not set]";
        s << "\t";

        if(o.description)
            s << endl << o.description;

        return s;
    }
};

/**
  * Environment class used for reading command line params and environemnt file, and manage
  * options in them. Each option must be registered in constructor of this class.
  * @name Environment
  **/
class Environment
{
private:
    //////////////////////////
    /** @section Variables **/

    int maxOptions;                             //< Maximal number of options
    int numParams;                              //< Number of columns in parameter table
    int paramRows;                              //< Number of rows in parameter table
    char* optionalParams;                       //< String with prefixes with non-optional parameters
    int curRow;                                 //< Current row in parameter table
    char*** params;                             //< Parameter table. 2D array of strings, first column are parameters, next are options
                                                //< prefixed by char passed to function GetCmdlineParams paired with corresponding parameters.
    int numOptions;                             //< Number of registered options
    Option* options;                            //< Options table

    ////////////////////////
    /** @section Methods **/

    /**
      * Method for checking variable type.
      * @name CheckVariableType
      * @param const char* - Value string
      * @param const OptionType - wanted variable type
      * @return If type matches, then true, otherwise false
      **/
    bool CheckVariableType(const char* value, const OptionType type) const;

    /**
      * Method for parsing environment file. Gets string from buffer, skipping leading whitespaces,
      * and appends it at the end of string parameter. Returns pointer next to string in the buffer
      * with trailing whitespaces skipped. If no string in the buffer can be found, returns NULL
      * @name ParseString
      * @param char* - Input line for parsing
      * @param char* - Buffer to append gathered string to
      * @return Pointer next to string or NULL if unsuccessful
      **/
    char* ParseString(char* buffer, char* str) const;

    /**
      * Method for parsing boolean value out of value string, e.g. "true" to (bool)true, and "false"
      * to (bool)false.
      * @name ParseBool
      * @param const char* - String containing boolean value
      * @return Boolean value
      **/
    bool ParseBool(const char* valueString) const;

    /**
      * Method for finding option by name.
      * @name FindOption
      * @param const char* - name of option to search
      * @param const bool - whether option is critical for application (defaults false)
      * @return Option id in array
      **/
    int FindOption(const char* name, const bool isFatal = false) const;

public:
    /**
      * Sets singleton pointer to one passed by argument
      * @name SetSingleton
      * @param Environment* - new singleton pointer
      * @return None
      **/
    static Environment* GetSingleton();

    /**
      * Gets singleton pointer, if NULL - then exit
      * @name GetSingleton
      * @param None
      * @return Environment*
      **/
    static void SetSingleton(Environment* e);

    /**
      * Deletes singleton pointer
      * @name DeleteSingleton
      * @param None
      * @return None
      **/
    static void DeleteSingleton();
    
    /**
      * Prints out all environment variable names
      * @name PrintUsage
      * @param ostream& - Stream to output
      * @return None
      **/
    virtual void PrintUsage(ostream &s) const;
    
    /**
      * Gets global option values 
      * @name SetStaticOptions
      * @param None
      * @return None
      **/
    virtual void SetStaticOptions();

	bool Parse(const int argc, char **argv, bool useExePath, char* overridePath = NULL, const char* overrideDefault = "default.env");

	void CodeString(char *buff, int max);

	void DecodeString(char *buff, int max);

	void SaveCodedFile(char *filenameText, char *filenameCoded);

	virtual void RegisterOptions() = 0;

    /** Method for registering new option.
      * Using this method is possible to register new option with it's name, type,
      * abbreviation and default value.
      * @name RegisterOption
      * @param name Name of the option.
      * @param type The type of the variable.
      * @param abbrev If not NULL, alias usable as a command line option.
      * @param defValue Implicit value used if not specified explicitly.
      * @return None
      **/
    void RegisterOption(const char *name, const OptionType type, const char *abbrev, const char *defValue = NULL);
    
    /** Method for setting new Int.
      * @name SetInt
      * @param name Name of the option.
      * @param value Value of the option
      * @return None
      **/
    void SetInt(const char *name, const int value);
    
    /** Method for setting new Float.
      * @name SetFloat
      * @param name Name of the option.
      * @param value Value of the option
      * @return None
      **/
    void SetFloat(const char *name, const float value);
    
    /** Method for setting new Bool.
      * @name SetBool
      * @param name Name of the option.
      * @param value Value of the option
      * @return None
      **/
    void SetBool(const char *name, const bool value);
    
    /** Method for setting new String.
      * @name SetString
      * @param name Name of the option.
      * @param value Value of the option
      * @return None
      **/
    void SetString(const char *name, const char *value);
    
    /** Method for getting Bool
      * @name GetBool
      * @param const char* - Name of the option.
      * @param const bool - whether is this value critical
      * @return the Bool
      **/
    bool GetBool(const char *name, const bool isFatal = false) const;
  
    /** Method for getting Int
      * @name GetInt
      * @param const char* - Name of the option.
      * @param const bool - whether is this value critical
      * @return the Int
      **/
    int GetInt(const char *name,const bool isFatal = false) const;
    
    /** Method for getting Float
      * @name GetFloat
      * @param const char* - Name of the option.
      * @param const bool - whether is this value critical
      * @return the Float
      **/
    float GetFloat(const char *name,const bool isFatal = false) const;
    
    /** Method for getting Double
      * @name GetDouble
      * @param const char* - Name of the option.
      * @param const bool - whether is this value critical
      * @return the Double
      **/
    double GetDouble(const char *name,const bool isFatal = false) const;
    
    /** Get Integer value
      * @name GetIntValue
      * @param const char* - Value name
      * @param int& - Place to store variable
      * @param const bool - Whether value is critical
      * @return bool - Whether succeeded
      **/
    bool GetIntValue(const char *name, int &value, const bool isFatal = false) const;
  
    /** Get Double value
      * @name GetDoubleValue
      * @param const char* - Value name
      * @param double& - Place to store variable
      * @param const bool - Whether value is critical
      * @return bool - Whether succeeded
      **/
    bool GetDoubleValue(const char *name, double &value, const bool isFatal = false) const;
    
    /** Get Float value
      * @name GetFloatValue
      * @param const char* - Value name
      * @param float& - Place to store variable
      * @param const bool - Whether value is critical
      * @return bool - Whether succeeded
      **/
    bool GetFloatValue(const char *name, float &value, const bool isFatal = false) const;
    
    /** Get Bool value
      * @name GetBoolValue
      * @param const char* - Value name
      * @param bool& - Place to store variable
      * @param const bool - Whether value is critical
      * @return bool - Whether succeeded
      **/
    bool GetBoolValue(const char *name, bool &value, const bool isFatal = false) const;
    
    /** Get Char* value
      * @name GetStringValue
      * @param const char* - Value name
      * @param char* - Place to store variable
      * @param const bool - Whether value is critical
      * @return bool - Whether succeeded
      **/
    bool GetStringValue(const char *name, char *value, const bool isFatal = false) const;
    
    /** Get String value
      * @name GetStringValue
      * @param const char* - Value name
      * @param string& - Place to store variable
      * @param const bool - Whether value is critical
      * @return bool - Whether succeeded
      **/
    bool GetStringValue(const char *name, string &value, const bool isFatal = false) const; 
    
    /** Check if the specified switch is present in the command line.
      * Primary use of this function is to check for presence of some special
      * switch, e.g. -h for help. It can be used anytime.
      * @name CheckForSwitch
      * @param argc Number of command line arguments.
      * @param argv Array of command line arguments.
      * @param swtch Switch we are checking for.
      * @return true if found, false elsewhere.
      **/
    bool CheckForSwitch(const int argc, char **argv, const char swtch) const;
    
    /** First pass of parsing command line.
      * This function parses command line and gets from there all non-optional
      * parameters (i.e. not prefixed by the dash) and builds a table of
      * parameters. According to param optParams, it also writes some optional
      * parameters to this table and pair them with corresponding non-optional
      * parameters.
      * @name ReadCmdlineParams
      * @param argc Number of command line arguments.
      * @param argv Array of command line arguments.
      * @param optParams String consisting of prefixes to non-optional parameters.
      *                  All prefixes must be only one character wide !
      * @return None
      **/
    void ReadCmdlineParams(const int argc, char **argv, const char *optParams);
    
    /** Reading of the environment file.
      * This function reads the environment file.
      * @name ReadEnvFile
      * @param filename The name of the environment file to read.
      * @return true if OK, false in case of error.
      **/
    bool ReadEnvFile(const char *filename);

    /** Second pass of the command line parsing.
      * Performs parsing of the command line ignoring all parameters and options
      * read in the first pass. Builds table of options.
      * @param argc Number of command line arguments.
      * @param argv Array of command line arguments.
      * @param index Index of non-optional parameter, of which options should
      * be read in.
      **/
    void ParseCmdline(const int argc, char **argv, const int index);
    
    /** Parameters number query function.
      * This function returns number of non-optional parameters specified on
      * the command line.
      **/
    int GetParamNum() const { return paramRows; }

    /** Returns the indexed parameter or corresponding optional parameter.
      * This function is used for queries on individual non-optional parameters.
      * The table must be constructed to allow this function to operate (i.e. the
      * first pass of command line parsing must be done).
      * @param name If space (' '), returns the non-optional parameter, else returns
      *             the corresponding optional parameter.
      * @param index Index of the non-optional parameter to which all other options
      *             corresponds.
      * @param value Return value of the queried parameter.
      * @return true if OK, false in case of error or if the parameter wasn't
      *         specified.
      **/
    bool GetParam(const char name, const int index, char *value) const;
    
    /** Determine whether option is present.
      * @name OptionPresent
      * @value const char* - Specifies name of option
      * @return True if present, otherwise false
      **/
    bool OptionPresent(const char *name) const;
    
    static Environment* mEnvironment;           //< Singleton pointer to Environment instance

public:
    /////////////////////////////
    /** @section Constructors **/
    
    /**
      * Default constructor
      * @name Environment
      * @param None
      * @return None
      **/
    Environment();

    ////////////////////////////
    /** @section Destructors **/
    
    /**
      * Default destructor, virtual
      * @name Environment
      * @param None
      * @return None
      **/
    virtual ~Environment();
};
  
#endif