import { Test, TestingModule } from '@nestjs/testing';
import { ImageRetrievalController } from './image-retrieval.controller';

describe('ImageRetrievalController', () => {
  let controller: ImageRetrievalController;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [ImageRetrievalController],
    }).compile();

    controller = module.get<ImageRetrievalController>(ImageRetrievalController);
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });
});
