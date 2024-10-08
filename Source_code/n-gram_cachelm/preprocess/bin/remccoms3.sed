#! /bin/sed -nf

# Remove C and C++ comments, by Brian Hiles (brian_hiles@rocketmail.com)

# Sped up (and bugfixed to some extent) by Paolo Bonzini (bonzini@gnu.org)
# Works its way through the line, copying to hold space the text up to the
# first special character (/, ", ').  The original version went exactly a
# character at a time, hence the greater speed of this one.  But the concept
# and especially the trick of building the line in hold space are entirely
# merit of Brian.


:loop

# This line is sufficient to remove C++ comments!
/^\/\// s,.*,,

/^$/{
  x
  p
  n
  b loop
}
/^"/{
  :double
  /^$/{
    x
    p
    n
    /^"/b break
    b double
  }

  H
  x
  s,\n\(.[^\"]*\).*,\1,
  x
  s,.[^\"]*,,
  
  /^"/b break
  /^\\/{
    H
    x
    s,\n\(.\).*,\1,
    x
    s/.//
  }
  b double
}

/^'/{
  :single
  /^$/{
    x
    p
    n
    /^'/b break
    b single
  }
  H
  x
  s,\n\(.[^\']*\).*,\1,
  x
  s,.[^\']*,,
  
  /^'/b break
  /^\\/{
    H
    x
    s,\n\(.\).*,\1,
    x
    s/.//
  }
  b single
}

/^\/\*/{
  s/.//
  :ccom
  s,^.[^*]*,,
  /^$/ n
  /^\*\//{
    s/..//
    b loop
  }
  b ccom
}

:break
H
x
s,\n\(.[^"'/]*\).*,\1,
x
s/.[^"'/]*//
b loop
