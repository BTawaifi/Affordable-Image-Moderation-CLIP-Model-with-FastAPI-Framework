export class FlaggingMicroserviceException extends Error {
    constructor(message: string) {
      super(message);
      this.name = 'FlaggingMicroserviceException';
    }
  }
  