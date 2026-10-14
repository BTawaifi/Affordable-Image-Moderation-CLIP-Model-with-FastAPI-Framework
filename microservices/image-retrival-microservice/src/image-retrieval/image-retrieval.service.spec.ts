import { Test, TestingModule } from '@nestjs/testing';
import { ImageRetrievalService } from './image-retrieval.service';

describe('ImageRetrievalService', () => {
  let service: ImageRetrievalService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [ImageRetrievalService],
    }).compile();

    service = module.get<ImageRetrievalService>(ImageRetrievalService);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });
});
