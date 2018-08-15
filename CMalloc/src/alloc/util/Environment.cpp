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

#include "Environment.h"

#define strcasecmp _stricmp
#define strncasecmp _strnicmp
#define MAX_STRING_LEN 65536

Environment* Environment::mEnvironment = NULL;

char *GetPath(const char *s)
{
  int i=strlen(s);
  for (; i>0; i--) {
    if (s[i]=='/' || s[i]=='\\')
      break;
  }
  
  char *path = new char[i+1];
  int j = 0;
  for (; j<i; j++)
    path[j] = s[j];
  path[j] = 0;
  return path;
}

/**
  * Method for checking variable type.
  * @name CheckVariableType
  * @param const char* - Value string
  * @param const OptionType - wanted variable type
  * @return If type matches, then true, otherwise false
  **/
bool Environment::CheckVariableType(const char* value, const OptionType type) const
{
    char *t;

    switch(type)
    {
    case optInt:
        // From Minimax - TODO: Check whether works!
        strtol(value, &t, 10);
        if(value + strlen(value) != t)
            return false;
        else
            return true;
        break;

    case optFloat:
        // From Minimax - TODO: Check whether works!
        strtod(value, &t);
        if(value + strlen(value) != t)
            return false;
        else
            return true;
        break;

    case optBool:
        // If string contains either 'true' or 'false' it's valid
        if( !strcasecmp(value, "true") ||
            !strcasecmp(value, "false"))
            return true;
        else
            return false;
        break;

    case optString:
        // String is always a string
        return true;
        break;
    }

    return false;
}

/**
  * Method for parsing environment file. Gets string from buffer, skipping leading whitespaces,
  * and appends it at the end of string parameter. Returns pointer next to string in the buffer
  * with trailing whitespaces skipped. If no string in the buffer can be found, returns NULL
  * @name ParseString
  * @param char* - Input line for parsing
  * @param char* - Buffer to append gathered string to
  * @return Pointer next to string or NULL if unsuccessful
  **/
char* Environment::ParseString(char* buffer, char* str) const
{
    char *s = buffer;
    char *t = str + strlen(str);

    // Skip leading whitespaces
    while (*s == ' ' || *s == '\t' || *s == '\n' || *s == 0x0d)
        s++;

    if (*s == 0)
        return NULL;

    while ( (*s >= 'a' && *s <= 'z') ||
            (*s >= 'A' && *s <= 'Z') ||
            (*s >= '0' && *s <= '9') ||
            (*s == '_'))
        *t++ = *s++;

    *t = 0;

    // Skip trailing whitespaces
    while (*s == ' ' || *s == '\t' || *s == '\n' || *s == 0x0d)
        s++;

    return s;
}

/**
  * Method for parsing boolean value out of value string, e.g. "true" to (bool)true, and "false"
  * to (bool)false.
  * @name ParseBool
  * @param const char* - String containing boolean value
  * @return Boolean value
  **/
bool Environment::ParseBool(const char* valueString) const
{
    bool value = true;
  
    if (!strcasecmp(valueString, "false"))
        value = false;
  
    return value;
}

/**
  * Method for finding option by name.
  * @name FindOption
  * @param const char* - name of option to search
  * @param const bool - whether option is critical for application (defaults false)
  * @return Option id in array
  **/
int Environment::FindOption(const char* name, const bool isFatal) const
{
    int i;
    bool found = false;

    // Is this option registered ?
    for (i = 0; i < numOptions; i++)
    {
        if (!strcmp(options[i].name, name)) 
        {
            found = true;
            break;
        }
    }

    if (!found) 
    {
        // No registration found - TODO: Log it!!!
        //Debug << "Internal error: Required option " << name << " not registered.\n" << flush;
        exit(1);
    }

    if (options[i].value == NULL && options[i].defaultValue == NULL)
    {
        // This option was not initialised to some value
        // TODO: Do the logging
        if (isFatal) 
        {
            //Debug << "Error: Required option " << name << " not found.\n" << flush;
            exit(1);
        }
        else 
        {
            //Debug << "Error: Required option " << name << " not found.\n" << flush;
            return -1;
        }
    }

    return i;
}

/**
  * Sets singleton pointer to one passed by argument
  * @name SetSingleton
  * @param Environment* - new singleton pointer
  * @return None
  **/
void Environment::SetSingleton(Environment *e)
{
    mEnvironment = e;
}

/**
  * Gets singleton pointer, if NULL - then exit
  * @name GetSingleton
  * @param None
  * @return Environment*
  **/
Environment* Environment::GetSingleton()
{
    if (!mEnvironment) 
    {
        cerr << "Environment not allocated!!";
        exit(1);
    }
  
    return mEnvironment;
}

/**
  * Deletes singleton pointer
  * @name DeleteSingleton
  * @param None
  * @return None
  **/
void Environment::DeleteSingleton()
{
    if(mEnvironment)
        delete mEnvironment;
}

/**
  * Prints out all environment variable names
  * @name PrintUsage
  * @param ostream& - Stream to output
  * @return None
  **/
void Environment::PrintUsage(ostream &s) const
{
    s << "Registered options:\n";
    for (int j = 0; j < numOptions; j++)
        s << options[j] << "\n";
    s << flush;
}

/**
  * Gets global option values 
  * @name SetStaticOptions
  * @param None
  * @return None
  **/
void Environment::SetStaticOptions()
{
}

bool Environment::Parse(const int argc, char **argv, bool useExePath, char* overridePath, const char* overrideDefault)
{
  bool result = true;
  // Read the names of the scene, environment and output files
  ReadCmdlineParams(argc, argv, "");

  char *envFilename = new char[128];
  char filename[64];

  // Get the environment file name
  if (!GetParam(' ', 0, filename)) {
    // user didn't specify environment file explicitly, so
    strcpy(filename, overrideDefault);
  }
  cout << "Using environment file: " << filename << endl;
  
  if (useExePath) {
    char *path = GetPath(argv[0]);
    if (*path != 0)
      sprintf(envFilename, "%s/%s", path, filename);
    else
      strcpy(envFilename, filename);
    
    delete path;
  }
  else if(overridePath != NULL) {
	  sprintf(envFilename, "%s/%s", overridePath, filename);
  }
  else
    strcpy(envFilename, filename);

  
  // Now it's time to read in environment file.
  if (!ReadEnvFile(envFilename)) {
    // error - bad input file name specified ?
    cerr<<"Error parsing environment file "<<envFilename<<endl;
	result = false;
  }
  delete [] envFilename;

  // Parse the command line; options given on the command line subsume
  // stuff specified in the input environment file.
  ParseCmdline(argc, argv, 0);

  SetStaticOptions();

  // Check for request for help
  if (CheckForSwitch(argc, argv, '?')) {
    PrintUsage(cout);
    exit(0);
  }
  
  return true;
}

const char code[] = "JIDHipewhfdhyd74387hHO&{WK:DOKQEIDKJPQ*H#@USX:#FWCQ*EJMQAHPQP(@G#RD";

void Environment::DecodeString(char *buff, int max)
{
  buff[max] = 0;
  char *p = buff;
  const char *cp = code; 
  for (; *p; p++) {
    if (*p != '\n')
      *p = *p ^ *cp;
    ++cp;
    if (*cp == 0)
      cp = code;
  }
}

void Environment::CodeString(char *buff, int max)
{
  buff[max] = 0;
  char *p = buff;
  const char *cp = code; 
  for (; *p; p++) {
    if (*p != '\n')
      *p = *p ^ *cp;
    ++cp;
    if (*cp == 0)
      cp = code;
  }
}

void Environment::SaveCodedFile(char *filenameText, char *filenameCoded)
{
  ifstream envStream(filenameText);
  
  // some error had occured
  if (envStream.fail()) {
    cerr << "Error: Can't open file " << filenameText << " for reading (err. "
         << envStream.rdstate() << ").\n";
    return;
  }

  char buff[256];
  envStream.getline(buff, 255);
  buff[8] = 0;
  if (strcmp(buff, "CGX_CF10") == 0)
    return;
  
  ofstream cStream(filenameCoded);
  cStream<<"CGX_CF10";
  
  // main loop
  for (;;) {
    // read in one line
    envStream.getline(buff, 255);
    if (!envStream)
      break;
    CodeString(buff, 255);
    cStream<<buff;
  }
  
}

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
void Environment::RegisterOption(const char *name, const OptionType type, const char *abbrev, const char *defValue)
{
    int i;

    // Make sure this option was not yet registered
    for (i = 0; i < numOptions; i++)
    {
        if (!strcmp(name, options[i].name)) 
        {
          //Debug << "Error: Option " << name << " registered twice.\n";
          exit(1);
        }
    }

    // Make sure we have enough room in memory
    if (numOptions >= maxOptions) 
    {
        //Debug << "Error: Too many options. Try enlarge the maxOptions " << "definition.\n";
        exit(1);
    }

    // Make sure the abbreviation doesn't start with 'D'
    if (abbrev != NULL && (abbrev[0] == 'D' )) 
    {
        //Debug << "Internal error: reserved switch " << abbrev << " used as an abbreviation.\n";
        exit(1);
    }

    // new option
    options[numOptions].type = type;
    options[numOptions].name = strdup(name);

    // assign abbreviation, if requested
    if (abbrev != NULL) 
    {
        options[numOptions].abbrev = strdup(abbrev);
    }
  
    // assign default value, if requested
    if (defValue != NULL) 
    {
        options[numOptions].defaultValue = strdup(defValue);
        if (!CheckVariableType(defValue, type)) 
        {
            //Debug << "Internal error: Inconsistent type and default value in option " << name << ".\n";
            exit(1);
        }
    }

    // new option registered
    numOptions++;
}

/** Method for setting new Int.
  * @name SetInt
  * @param name Name of the option.
  * @param value Value of the option
  * @return None
  **/
void Environment::SetInt(const char *name, const int value)
{
    int i = FindOption(name);

    if (i<0)
        return;

    if (options[i].type == optInt) 
    {
        delete [] options[i].value;
        options[i].value = new char[16];
        sprintf(options[i].value, "%.15d", value);
    }
    else 
    {
        //Debug << "Internal error: Trying to set non-integer option " << name << " to integral value.\n" << flush;
        exit(1);
    }
}
    
/** Method for setting new Float.
  * @name SetFloat
  * @param name Name of the option.
  * @param value Value of the option
  * @return None
  **/
void Environment::SetFloat(const char *name, const float value)
{
    int i = FindOption(name);

    if (i<0)
        return;

    if (options[i].type == optFloat) 
    {
        delete [] options[i].value;
        options[i].value = new char[25];
        sprintf(options[i].value, "%.15e", value);
    }
    else 
    {
        //Debug << "Internal error: Trying to set non-float option " << name << " to float value.\n" << flush;
        exit(1);
    }
}
    
/** Method for setting new Bool.
  * @name SetBool
  * @param name Name of the option.
  * @param value Value of the option
  * @return None
  **/
void Environment::SetBool(const char *name, const bool value)
{
    int i = FindOption(name);

    if (i<0)
        return;

    if (options[i].type == optBool) 
    {
        delete [] options[i].value;

        options[i].value = new char[6];

        if (value)
            sprintf(options[i].value, "true");
        else
            sprintf(options[i].value, "false");
    }
    else 
    {
        //Debug << "Internal error: Trying to set non-bool option " << name << " to boolean value.\n" << flush;
        exit(1);
    }
}
    
/** Method for setting new String.
  * @name SetString
  * @param name Name of the option.
  * @param value Value of the option
  * @return None
  **/
void Environment::SetString(const char *name, const char *value)
{
    int i = FindOption(name);

    if (i<0)
        return;

    if (options[i].type == optString) 
    {
        delete [] options[i].value;
        options[i].value = ::strdup(value);
    }
    else 
    {
        //Debug << "Internal error: Trying to set non-string option " << name << " to string value.\n" << flush;
        exit(1);
    }
}

/** Method for getting Bool
  * @name GetBool
  * @param const char* - Name of the option.
  * @param const bool - whether is this value critical
  * @return the Bool
  **/
bool Environment::GetBool(const char *name, const bool isFatal) const
{
    bool ret;
    if (GetBoolValue(name, ret, isFatal))
        return ret;
    else
        return false;
}
  
/** Method for getting Int
  * @name GetInt
  * @param const char* - Name of the option.
  * @param const bool - whether is this value critical
  * @return the Int
  **/
int Environment::GetInt(const char *name,const bool isFatal) const
{
    int ret;
    if (GetIntValue(name, ret, isFatal))
        return ret;
    else
    {
        cerr << "Error: GetInt value not found!";
        exit(-1);
    }
}
    
/** Method for getting Float
  * @name GetFloat
  * @param const char* - Name of the option.
  * @param const bool - whether is this value critical
  * @return the Float
  **/
float Environment::GetFloat(const char *name, const bool isFatal) const
{
    float ret;
    if (GetFloatValue(name, ret, isFatal))
        return ret;
    else
    {
	    cerr << "Error: GetFloat value not found!";
	    exit(-1);
    }
}
    
/** Method for getting Double
  * @name GetDouble
  * @param const char* - Name of the option.
  * @param const bool - whether is this value critical
  * @return the Double
  **/
double Environment::GetDouble(const char *name,const bool isFatal) const
{
    double ret;
    if (GetDoubleValue(name, ret, isFatal))
        return ret;
    else
    {
	    cerr << "Error: GetDouble value not found!";
	    exit(-1);
    }
}

/** Get Integer value
  * @name GetIntValue
  * @param const char* - Value name
  * @param int& - Place to store variable
  * @param const bool - Whether value is critical
  * @return bool - Whether succeeded
  **/
bool Environment::GetIntValue(const char *name, int &value, const bool isFatal) const
{
    int i = FindOption(name, isFatal);

    if (i<0)
        return false;

    if (options[i].value != NULL) 
    {
        value = strtol(options[i].value, NULL, 10);
    } 
    else 
    {
        value = strtol(options[i].defaultValue, NULL, 10);
    }

    return true;
}
  
/** Get Double value
  * @name GetDoubleValue
  * @param const char* - Value name
  * @param double& - Place to store variable
  * @param const bool - Whether value is critical
  * @return bool - Whether succeeded
  **/
bool Environment::GetDoubleValue(const char *name, double &value, const bool isFatal) const
{
    int i = FindOption(name, isFatal);

    if (i<0)
        return false;

    if (options[i].value != NULL) 
    {
        value = strtod(options[i].value, NULL);
    }
    else 
    {
        value = strtod(options[i].defaultValue, NULL);
    }

    return true;
}
    
/** Get Float value
  * @name GetFloatValue
  * @param const char* - Value name
  * @param float& - Place to store variable
  * @param const bool - Whether value is critical
  * @return bool - Whether succeeded
  **/
bool Environment::GetFloatValue(const char *name, float &value, const bool isFatal) const
{
    int i = FindOption(name, isFatal);

    if (i<0)
        return false;

    if (options[i].value != NULL) 
    {
        value = (float)strtod(options[i].value, NULL);
    } 
    else 
    {
        value = (float)strtod(options[i].defaultValue, NULL);
    }

    return true;
}
    
/** Get Bool value
  * @name GetBoolValue
  * @param const char* - Value name
  * @param bool& - Place to store variable
  * @param const bool - Whether value is critical
  * @return bool - Whether succeeded
  **/
bool Environment::GetBoolValue(const char *name, bool &value, const bool isFatal) const
{
    int i = FindOption(name, isFatal);

    if (i < 0)
        return false;
  
    if (options[i].value != NULL) 
        value = ParseBool(options[i].value);
    else
        value = ParseBool(options[i].defaultValue);

    return true;
}
    
/** Get Char* value
  * @name GetStringValue
  * @param const char* - Value name
  * @param char* - Place to store variable
  * @param const bool - Whether value is critical
  * @return bool - Whether succeeded
  **/
bool Environment::GetStringValue(const char *name, char *value, const bool isFatal) const
{
    int i = FindOption(name, isFatal);

    if (i<0)
        return false;

    if (options[i].value != NULL)
        strcpy(value, options[i].value);
    else
        strcpy(value, options[i].defaultValue);

    return true;
}
    
/** Get String value
  * @name GetStringValue
  * @param const char* - Value name
  * @param string& - Place to store variable
  * @param const bool - Whether value is critical
  * @return bool - Whether succeeded
  **/
bool Environment::GetStringValue(const char *name, string &value, const bool isFatal) const
{
    char buffer[MAX_STRING_LEN];
    bool result = GetStringValue(name, buffer, isFatal);
    if (result)
        value = buffer;
    return result;
}

/** Check if the specified switch is present in the command line.
  * Primary use of this function is to check for presence of some special
  * switch, e.g. -h for help. It can be used anytime.
  * @name CheckForSwitch
  * @param argc Number of command line arguments.
  * @param argv Array of command line arguments.
  * @param swtch Switch we are checking for.
  * @return true if found, false elsewhere.
  **/
bool Environment::CheckForSwitch(const int argc, char **argv, const char swtch) const
{
    for (int i = 1; i < argc; i++)
        if ((argv[i][0] == '-') && (argv[i][1] == swtch))
            return true;
    return false;
}

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
void Environment::ReadCmdlineParams(const int argc, char **argv, const char *optParams)
{
    int i;

    // Make sure we are called for the first time
    if (optionalParams != NULL)
        return;

    numParams = (int)strlen(optParams) + 1;
    optionalParams = new char[numParams];
    strcpy(optionalParams, optParams);

    // First, count all non-optional parameters on the command line
    for (i = 1; i < argc; i++)
        if (argv[i][0] != '-')
            paramRows++;

    // if there is no non-optional parameter add a default one...
    if (paramRows == 0)
        paramRows = 1;
  
    // allocate and initialize the table for parameters
    params = new char **[numParams];
    for (i = 0; i < numParams; i++) 
    {
        params[i] = new char *[paramRows];
        for (int j = 0; j < paramRows; j++)
            params[i][j] = NULL;
    }

    // Now read all non-optional and optional parameters into the table
    curRow = -1;
    for (i = 1; i < argc; i++) 
    {
        if (argv[i][0] != '-') 
        {
            // non-optional parameter encountered
            curRow++;
            params[0][curRow] = new char[strlen(argv[i]) + 1];
            strcpy(params[0][curRow], argv[i]);
        }
        else
        {
            // option encountered
            char *t = strchr(optionalParams, argv[i][1]);
            if (t != NULL)
            {
                // this option is optional parameter
                int index = t - optionalParams + 1;
                if (curRow < 0)
                {
                    // it's a global parameter
                    for (int j = 0; j < paramRows; j++) 
                    {
                        params[index][j] = new char[strlen(argv[i] + 2) + 1];
                        strcpy(params[index][j], argv[i] + 2);
                    }
                }
                else 
                {
                    // it's a scene parameter
                    if (params[index][curRow] != NULL)
                    {
                        delete [] params[index][curRow];
                    }
                    params[index][curRow] = new char[strlen(argv[i] + 2) + 1];
                    strcpy(params[index][curRow], argv[i] + 2);
                }
            }
        }
    }

    curRow = 0;
}

/** Reading of the environment file.
  * This function reads the environment file.
  * @name ReadEnvFile
  * @param filename The name of the environment file to read.
  * @return true if OK, false in case of error.
  **/
bool Environment::ReadEnvFile(const char *filename)
{
    char buff[MAX_STRING_LEN], name[MAX_STRING_LEN];
    char *s, *t;
    int i, line = 0;
    bool found;
    //  igzstream envStream(envFilename);
    ifstream envStream(filename);

    // some error had occured
    if (envStream.fail())
    {
        cerr << "Error: Can't open file " << filename << " for reading (err. " << envStream.rdstate() << ").\n";
        return false;
    }

    name[0] = '\0';

    // main loop
    for (;;) 
    {
        // read in one line
        envStream.getline(buff, MAX_STRING_LEN-1);
    
        if (!envStream)
            break;

        line++;
        // get rid of comments
        s = strchr(buff, '#');
        if (s != NULL)
            *s = '\0';

        // get one identifier
        s = ParseString(buff, name);
	
        // parse line
        while (s != NULL) 
        {
            // it's a group name - make the full name
            if (*s == '{') 
            {
                strcat(name, ".");
                s++;
                s = ParseString(s, name);
                continue;
            }
        
            // end of group
            if (*s == '}') 
            {
                if (strlen(name) == 0) 
                {
                    cerr << "Error: unpaired } in " << filename << " (line " << line << ").\n";
                    envStream.close();
                    return false;
                }

                name[strlen(name) - 1] = '\0';
                t = strrchr(name, '.');

                if (t == NULL)
                    name[0] = '\0';
                else
                    *(t + 1) = '\0';

                s++;
                s = ParseString(s, name);

                continue;
            }

            // find variable name in the table
            found = false;
            for (i = 0; i < numOptions; i++)
            {
                if (!strcmp(name, options[i].name)) 
                {
                    found = true;
                    break;
                }
            }

            if (!found) 
            {
                cerr << "Warning: unknown option " << name << " in environment file " << filename << " (line " << line << ").\n";
            }
            else
            {
                switch (options[i].type) 
                {
                case optInt: 
                {
                    strtol(s, &t, 10);
                    if (t == s || (*t != ' ' && *t != '\t' && *t != '\0' && *t != '}')) 
                    {
                        cerr << "Error: Mismatch in int variable " << name << " in " << "environment file " << filename << " (line " << line << ").\n";
                        envStream.close();
                        return false;
                    }
                    
                    if (options[i].value != NULL)
                        delete [] options[i].value;

                    options[i].value = new char[t - s + 1];
                    strncpy(options[i].value, s, t - s);
                    options[i].value[t - s] = '\0';
                    s = t;
                    break;
                }

            case optFloat: 
                {
                    strtod(s, &t);
                    if (t == s || (*t != ' ' && *t != '\t' && *t != '\0' && *t != '}')) 
                    {
                        cerr << "Error: Mismatch in float variable " << name << " in " << "environment file " << filename << " (line " << line << ").\n";
                        envStream.close();
                        return false;
                    }

                    if (options[i].value != NULL)
                        delete [] options[i].value;

                    options[i].value = new char[t - s + 1];
                    strncpy(options[i].value, s, t - s);
                    options[i].value[t - s] = '\0';
                    s = t;
                    break;
                }

            case optBool: 
                {
                    t = s;
                    while ( (*t >= 'a' && *t <= 'z') ||
                            (*t >= 'A' && *t <= 'Z') ||
                            *t == '+' || *t == '-')
                        t++;

                    if ((   (!strncasecmp(s, "true", t - s)  && t - s == 4) ||
                            (!strncasecmp(s, "false", t - s) && t - s == 5)) &&
                        (*t == ' ' || *t == '\t' || *t == '\0' || *t == '}')) 
                    {
                        if (options[i].value != NULL)
                            delete [] options[i].value;
                        options[i].value = new char[t - s + 1];
                        strncpy(options[i].value, s, t - s);
                        options[i].value[t - s] = '\0';
                        s = t;
                    }
                    else 
                    {
                        cerr << "Error: Mismatch in bool variable " << name << " in " << "environment file " << filename << " (line " << line << ").\n";
                        envStream.close();
                        return false;
                    }
                    break;
                }

            case optString: 
                {
                    if (options[i].value != NULL)
                        delete [] options[i].value;

                    options[i].value = new char[strlen(s) + 1];
                    strcpy(options[i].value, s);
                    s += strlen(s);
                    int last = strlen(options[i].value)-1;

                    if (options[i].value[last] == 0x0a || options[i].value[last] == 0x0d)
                        options[i].value[last] = 0;	

                    break;
                }

            default: 
                {
                    //Debug << "Internal error: Unknown type of option.\n" << flush;
                    exit(1);
                }
                }
            }

            // prepare the variable name for next pass
            t = strrchr(name, '.');

            if (t == NULL)
                name[0] = '\0';
            else
                *(t + 1) = '\0';

            // get next identifier
            s = ParseString(s, name);
        }
    }

    envStream.close();
    return true;
}

/** Second pass of the command line parsing.
  * Performs parsing of the command line ignoring all parameters and options
  * read in the first pass. Builds table of options.
  * @param argc Number of command line arguments.
  * @param argv Array of command line arguments.
  * @param index Index of non-optional parameter, of which options should
  * be read in.
  **/
void Environment::ParseCmdline(const int argc, char **argv, const int index)
{
    int curIndex = -1;

    for (int i = 1; i < argc; i++) 
    {
        // if this parameter is non-optional, skip it and increment the counter
        if (argv[i][0] != '-') 
        {
            curIndex++;
            continue;
        }

        // make sure to skip all non-optional parameters
        char *t = strchr(optionalParams, argv[i][1]);
        if (t != NULL)
        continue;

        // if we are in the scope of the current parameter, parse it
        if (curIndex == -1 || curIndex == index) 
        {
            if (argv[i][1] == 'D') 
            {
                // it's a full name definition
                bool found = false;
                int j;

                char *t = strchr(argv[i] + 2, '=');
                if (t == NULL) 
                {
                    //Debug << "Error: Missing '=' in option. "<< "Syntax is -D<name>=<value>.\n" << flush;
                    exit(1);
                }

                for (j = 0; j < numOptions; j++)
                {
                    if (!strncmp(options[j].name, argv[i] + 2, t - argv[i] - 2) && (unsigned)(t - argv[i] - 2) == strlen(options[j].name)) 
                    {
                        found = true;
                        break;
                    }
                }

                if (!found) 
                {
                //Debug << "Warning: Unregistered option " << argv[i] << ".\n" << flush;
                //  exit(1);
                }
    
                if (found) 
                {
                    if (!CheckVariableType(t + 1, options[j].type)) 
                    {
                        //Debug << "Error: invalid type of value " << t + 1 << " in option " << options[j].name << ".\n";
                        exit(1);
                    }

                    if (options[j].value != NULL)
                        delete [] options[j].value;

                    options[j].value = strdup(t + 1);
                }
            }
            else
            {
                // it's an abbreviation
                bool found = false;
                int j;
	
                for (j = 0; j < numOptions; j++)
                {
                    if (options[j].abbrev != NULL && !strncmp(options[j].abbrev, argv[i] + 1, strlen(options[j].abbrev))) 
                    {
                        found = true;
                        break;
                    }
                
                    if (!found) 
                    {
                        //Debug << "Warning: Unregistered option " << argv[i] << ".\n" << flush;
                        //          exit(1);
                    }
                    if (found) 
                    {
                        if (!CheckVariableType(argv[i] + 1 + strlen(options[j].abbrev), options[j].type)) 
                        {
                            //Debug << "Error: invalid type of value " << argv[i] + 1 + strlen(options[j].abbrev) << "in option " << options[j].name << ".\n";
                            exit(1);
                        }

                        if (options[j].value != NULL)
                            delete [] options[j].value;
    
                        options[j].value = strdup(argv[i] + 1 + strlen(options[j].abbrev));
                    }
                }
            }
        }
    }
}

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
bool Environment::GetParam(const char name, const int index, char *value) const
{
    int column;

    if (index >= paramRows || index < 0)
        return false;

    if (name == ' ')
        column = 0;
    else 
    {
        char *t = strchr(optionalParams, name);

        if (t == NULL)
            return false;

        column = t - optionalParams + 1;
    }

    if (params[column][index] == NULL)
        return false;

    strcpy(value, params[column][index]);

    return true;
}

/** Determine whether option is present.
  * @name OptionPresent
  * @value const char* - Specifies name of option
  * @return True if present, otherwise false
  **/
bool Environment::OptionPresent(const char *name) const
{
    bool found = false;
    int i;

    for (i = 0; i < numOptions; i++)
    {
        if (!strcmp(options[i].name, name)) 
        {
            found = true;
            break;
        }
    }

    if (!found) 
    {
        //Debug << "Internal error: Option " << name << " not registered.\n" << flush;
        exit(1);
    }

    if (options[i].value != NULL || options[i].defaultValue != NULL)
        return true;
    else
        return false;
}

/**
  * Default constructor
  * @name Environment
  * @param None
  * @return None
  **/
Environment::Environment()
{
    optionalParams = NULL;
    paramRows = 0;
    numParams = 0;
    params = NULL;
    maxOptions = 500;

  
    // this is maximal nuber of options.
    numOptions = 0;

    options = new Option[maxOptions];

    if (options == NULL ) 
    {
        //Debug << "Error: Memory allocation failed.\n";
        exit(1);
    }

    return;
}
    
/**
  * Default destructor, virtual
  * @name Environment
  * @param None
  * @return None
  **/
Environment::~Environment()
{
    int i, j;

    // delete the params structure
    for (i = 0; i < numParams; i++) 
    {
        for (j = 0; j < paramRows; j++)
        {
            if (params[i][j] != NULL)
                delete[] params[i][j];
        }

        if (params[i] != NULL)
            delete[] params[i];
    }

    if (params != NULL)
        delete[] params;
  
    if (options != NULL)
        delete [] options;
  
    if (optionalParams != NULL)
        delete optionalParams;
}