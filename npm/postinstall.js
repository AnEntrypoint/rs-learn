#!/usr/bin/env node
'use strict';
const { ensureBinary } = require('./install');

ensureBinary({ verbose: true }).catch(e => {
  console.error(`rs-learn: postinstall could not pre-fetch binary: ${e.message}`);
  console.error('rs-learn: will retry on first invocation');
  process.exit(0);
});
